//! Declared media limits, enforced at the router.
//!
//! A supplier may decline a modality, which exempts it from inspection. The
//! declaration is only worth anything if the endpoint actually refuses it:
//! otherwise a video request is forwarded to a worker that will attempt to
//! decode it, and the traffic we said we do not take is served anyway.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use openai_protocol::{
    chat::ChatMessage,
    chat::MessageContent,
    common::ContentPart,
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, StringOrContentParts,
    },
};

/// Content parts of a chat message; empty when the content is a bare string.
fn message_parts(message: &ChatMessage) -> &[ContentPart] {
    let content = match message {
        ChatMessage::Root { content, .. }
        | ChatMessage::System { content, .. }
        | ChatMessage::User { content, .. }
        | ChatMessage::Tool { content, .. }
        | ChatMessage::Developer { content, .. } => Some(content),
        ChatMessage::Assistant { content, .. } => content.as_ref(),
        // Text-only by construction; no parts to inspect.
        ChatMessage::Function { .. } => None,
    };
    match content {
        Some(MessageContent::Parts(parts)) => parts,
        _ => &[],
    }
}

/// Content parts of a `/v1/responses` input item; empty when it carries none.
fn input_item_parts(item: &ResponseInputOutputItem) -> &[ResponseContentPart] {
    match item {
        ResponseInputOutputItem::Message { content, .. } => content,
        ResponseInputOutputItem::SimpleInputMessage { content, .. } => match content {
            StringOrContentParts::Array(parts) => parts,
            StringOrContentParts::String(_) => &[],
        },
        // Reasoning, tool calls and tool outputs are text-only by
        // construction.
        _ => &[],
    }
}

/// Index of the first message carrying a video part, if any.
///
/// Reported as `messages[i]` so a caller can find the offending part rather
/// than bisecting a long conversation.
pub fn first_video_message(messages: &[ChatMessage]) -> Option<usize> {
    messages.iter().position(|message| {
        message_parts(message)
            .iter()
            .any(|part| matches!(part, ContentPart::VideoUrl { .. }))
    })
}

/// Number of image parts across all messages.
///
/// A count rather than a position: the image ceiling is spent per image, so a
/// request carrying three images takes three slots. Every turn counts,
/// assistant ones included: the worker handles an image wherever it sits in
/// the conversation.
pub fn count_image_parts(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .flat_map(message_parts)
        .filter(|part| matches!(part, ContentPart::ImageUrl { .. }))
        .count()
}

/// Index of the first `/v1/responses` input item carrying a video part.
///
/// Reported as `input[i]` to match the field the caller sent, the way the chat
/// path reports `messages[i]`.
pub fn first_video_input_item(input: &ResponseInput) -> Option<usize> {
    let items = match input {
        // A bare string cannot carry media.
        ResponseInput::Text(_) => return None,
        ResponseInput::Items(items) => items,
    };

    items.iter().position(|item| {
        input_item_parts(item).iter().any(|part| {
            matches!(
                part,
                ResponseContentPart::InputVideo { .. } | ResponseContentPart::VideoUrl { .. }
            )
        })
    })
}

/// Whether a `/v1/responses` part is an image.
///
/// `openai_protocol` does not model image parts: they deserialize to
/// `ResponseContentPart::Unknown` with the original object intact, so the
/// type tag is read from there. The chat spelling (`image_url`) counts along
/// with the Responses one (`input_image`), for the same reason the video
/// guard accepts both.
fn is_image_input_part(part: &ResponseContentPart) -> bool {
    match part {
        ResponseContentPart::Unknown(value) => matches!(
            value.get("type").and_then(serde_json::Value::as_str),
            Some("input_image" | "image_url")
        ),
        _ => false,
    }
}

/// Number of image parts across all `/v1/responses` input items.
///
/// The Responses counterpart of [`count_image_parts`].
pub fn count_image_input_parts(input: &ResponseInput) -> usize {
    let items = match input {
        // A bare string cannot carry media.
        ResponseInput::Text(_) => return 0,
        ResponseInput::Items(items) => items,
    };

    items
        .iter()
        .flat_map(input_item_parts)
        .filter(|part| is_image_input_part(part))
        .count()
}

// ---------------------------------------------------------------------------
// Image area, for the megapixel admission ceiling.
//
// Counting images is not enough to protect base-GPU memory: 200 thumbnails and
// 200 full-resolution photographs are the same count and ~14x apart in
// preprocessing cost. Measured on a B300 node, ~1000 MP in flight was clean and
// ~1250 MP exhausted the base GPU, so the ceiling is expressed in pixels.
//
// Dimensions come from the file header, not a decode: every format below states
// width and height in its first bytes, so a base64 payload is measured by
// decoding a short prefix rather than the image.
// ---------------------------------------------------------------------------

/// Decoded bytes inspected when looking for image dimensions.
///
/// Bounded on purpose: a crafted payload must not make admission expensive.
/// A JPEG whose SOF sits behind a large EXIF block gets one larger retry.
const HEADER_PROBE_BYTES: usize = 512;
const JPEG_PROBE_BYTES: usize = 96 * 1024;

/// Area charged to an image whose dimensions cannot be read.
///
/// Remote URLs are the case that matters: the router does not fetch them, so
/// their size is unknowable at admission. The default is the spec's own
/// `max_total_pixels` ceiling -- the largest area an image is allowed to
/// contribute -- because the failure this guards against is a wedged node,
/// and a refused request is recoverable where an exhausted base GPU is not.
///
/// Tunable (`--unmeasured-image-megapixels`) rather than fixed: if real traffic
/// turns out to be URL-heavy, charging every URL the maximum would shed
/// legitimate requests, and that is a one-flag correction rather than a build.
/// Watch `smg_image_rejected_total` split by reason before lowering it.
pub const UNMEASURED_IMAGE_PIXELS: u64 = 12_845_056;

/// The base64 payload of a `data:` URL, if this is one.
fn data_url_payload(url: &str) -> Option<&str> {
    const MARKER: &str = ";base64,";
    let rest = url.strip_prefix("data:")?;
    let at = rest.find(MARKER)?;
    Some(&rest[at + MARKER.len()..])
}

/// Decode at most `max_bytes` of a base64 payload.
///
/// Takes whole quads so the decoder never sees a partial group; a payload that
/// is not clean base64 yields `None` and is charged the default.
fn decode_prefix(payload: &str, max_bytes: usize) -> Option<Vec<u8>> {
    let quads = max_bytes.div_ceil(3) * 4;
    let take = quads.min(payload.len()) / 4 * 4;
    STANDARD.decode(payload.as_bytes().get(..take)?).ok()
}

fn png_dimensions(b: &[u8]) -> Option<(u64, u64)> {
    if b.len() < 24 || &b[..8] != b"\x89PNG\r\n\x1a\n" || &b[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(b[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(b[20..24].try_into().ok()?);
    Some((u64::from(w), u64::from(h)))
}

fn gif_dimensions(b: &[u8]) -> Option<(u64, u64)> {
    if b.len() < 10 || (&b[..6] != b"GIF87a" && &b[..6] != b"GIF89a") {
        return None;
    }
    let w = u16::from_le_bytes(b[6..8].try_into().ok()?);
    let h = u16::from_le_bytes(b[8..10].try_into().ok()?);
    Some((u64::from(w), u64::from(h)))
}

/// WebP in its three container flavours: extended, lossless, lossy.
fn webp_dimensions(b: &[u8]) -> Option<(u64, u64)> {
    if b.len() < 30 || &b[..4] != b"RIFF" || &b[8..12] != b"WEBP" {
        return None;
    }
    match &b[12..16] {
        // VP8X: 24-bit canvas width-1 / height-1, little endian.
        b"VP8X" => {
            let w = u32::from(b[24]) | u32::from(b[25]) << 8 | u32::from(b[26]) << 16;
            let h = u32::from(b[27]) | u32::from(b[28]) << 8 | u32::from(b[29]) << 16;
            Some((u64::from(w) + 1, u64::from(h) + 1))
        }
        // VP8L: 14 bits each, packed behind a 0x2f signature byte.
        b"VP8L" => {
            if b[20] != 0x2f {
                return None;
            }
            let bits = u32::from_le_bytes(b[21..25].try_into().ok()?);
            Some((u64::from(bits & 0x3FFF) + 1, u64::from(bits >> 14 & 0x3FFF) + 1))
        }
        // VP8 : key-frame header after the 0x9d 0x01 0x2a sync code.
        b"VP8 " => {
            if b[23..26] != [0x9d, 0x01, 0x2a] {
                return None;
            }
            let w = u16::from_le_bytes(b[26..28].try_into().ok()?) & 0x3FFF;
            let h = u16::from_le_bytes(b[28..30].try_into().ok()?) & 0x3FFF;
            Some((u64::from(w), u64::from(h)))
        }
        _ => None,
    }
}

/// Walk JPEG segments to the start-of-frame, which states the dimensions.
///
/// The marker is not at a fixed offset: APP segments (EXIF above all) sit in
/// front of it and can be tens of kilobytes long, so the header has to be
/// walked rather than indexed.
fn jpeg_dimensions(b: &[u8]) -> Option<(u64, u64)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 9 < b.len() {
        if b[i] != 0xFF || b[i + 1] == 0xFF {
            i += 1; // fill byte or desync
            continue;
        }
        let marker = b[i + 1];
        // Standalone markers carry no length field.
        if marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            i += 2;
            continue;
        }
        // SOF0-3, 5-7, 9-11, 13-15. C4/C8/CC are tables, not frames.
        if matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) {
            let h = u16::from_be_bytes(b[i + 5..i + 7].try_into().ok()?);
            let w = u16::from_be_bytes(b[i + 7..i + 9].try_into().ok()?);
            return Some((u64::from(w), u64::from(h)));
        }
        let length = usize::from(u16::from_be_bytes(b[i + 2..i + 4].try_into().ok()?));
        if length < 2 {
            return None;
        }
        i += 2 + length;
    }
    None
}

fn sniff_dimensions(b: &[u8]) -> Option<(u64, u64)> {
    png_dimensions(b)
        .or_else(|| gif_dimensions(b))
        .or_else(|| webp_dimensions(b))
        .or_else(|| jpeg_dimensions(b))
}

/// Pixel area of one image reference.
///
/// Anything unreadable -- a remote URL, an unknown container, a truncated or
/// corrupt payload -- is charged `UNMEASURED_IMAGE_PIXELS`. Erring high is the
/// safe direction: an under-charged image is one that can still exhaust the GPU.
pub fn image_url_pixels(url: &str, unmeasured: u64) -> u64 {
    let Some(payload) = data_url_payload(url) else {
        return unmeasured;
    };
    if let Some(head) = decode_prefix(payload, HEADER_PROBE_BYTES) {
        if let Some((w, h)) = sniff_dimensions(&head) {
            return w.saturating_mul(h);
        }
        // A JPEG hiding its SOF behind EXIF: one bounded second look.
        if head.len() >= 2 && head[0] == 0xFF && head[1] == 0xD8 {
            if let Some(more) = decode_prefix(payload, JPEG_PROBE_BYTES) {
                if let Some((w, h)) = jpeg_dimensions(&more) {
                    return w.saturating_mul(h);
                }
            }
        }
    }
    unmeasured
}

/// Total image area across all messages, in pixels.
pub fn sum_image_pixels(messages: &[ChatMessage], unmeasured: u64) -> u64 {
    messages
        .iter()
        .flat_map(message_parts)
        .filter_map(|part| match part {
            ContentPart::ImageUrl { image_url } => Some(image_url_pixels(&image_url.url, unmeasured)),
            _ => None,
        })
        .fold(0u64, u64::saturating_add)
}

/// Total image area across `/v1/responses` input items, in pixels.
pub fn sum_image_input_pixels(input: &ResponseInput, unmeasured: u64) -> u64 {
    let items = match input {
        ResponseInput::Text(_) => return 0,
        ResponseInput::Items(items) => items,
    };
    items
        .iter()
        .flat_map(input_item_parts)
        .filter(|part| is_image_input_part(part))
        .map(|part| {
            let ResponseContentPart::Unknown(value) = part else {
                return unmeasured;
            };
            // `image_url` is either a bare string or an object with `url`.
            value
                .get("image_url")
                .and_then(|v| {
                    v.as_str()
                        .or_else(|| v.get("url").and_then(serde_json::Value::as_str))
                })
                .or_else(|| value.get("url").and_then(serde_json::Value::as_str))
                .map_or(unmeasured, |u| image_url_pixels(u, unmeasured))
        })
        .fold(0u64, u64::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_protocol::common::{ImageUrl, VideoUrl};

    fn user_with(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Parts(parts),
            name: None,
        }
    }

    fn text_user(text: &str) -> ChatMessage {
        ChatMessage::User {
            content: MessageContent::Text(text.to_string()),
            name: None,
        }
    }

    #[test]
    fn plain_text_is_untouched() {
        assert_eq!(first_video_message(&[text_user("hello")]), None);
    }

    #[test]
    fn image_parts_are_untouched() {
        let message = user_with(vec![
            ContentPart::Text {
                text: "what is this".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.invalid/a.png".into(),
                    ..Default::default()
                },
            },
        ]);
        assert_eq!(first_video_message(&[message]), None);
    }

    #[test]
    fn video_is_found_and_located() {
        let messages = vec![
            text_user("first"),
            user_with(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.invalid/a.mp4".into(),
                    ..Default::default()
                },
            }]),
        ];
        assert_eq!(first_video_message(&messages), Some(1));
    }

    #[test]
    fn video_is_found_behind_a_leading_image() {
        let message = user_with(vec![
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.invalid/a.png".into(),
                    ..Default::default()
                },
            },
            ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.invalid/a.mp4".into(),
                    ..Default::default()
                },
            },
        ]);
        assert_eq!(first_video_message(&[message]), Some(0));
    }

    #[test]
    fn video_in_an_assistant_turn_still_counts() {
        let message = ChatMessage::Assistant {
            content: Some(MessageContent::Parts(vec![ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.invalid/a.mp4".into(),
                    ..Default::default()
                },
            }])),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        };
        assert_eq!(first_video_message(&[message]), Some(0));
    }

    fn simple_item(parts: Vec<ResponseContentPart>) -> ResponseInputOutputItem {
        ResponseInputOutputItem::SimpleInputMessage {
            content: StringOrContentParts::Array(parts),
            role: "user".into(),
            r#type: None,
        }
    }

    #[test]
    fn responses_plain_string_input_is_untouched() {
        assert_eq!(
            first_video_input_item(&ResponseInput::Text("hello".into())),
            None
        );
    }

    #[test]
    fn responses_text_parts_are_untouched() {
        let input = ResponseInput::Items(vec![simple_item(vec![ResponseContentPart::InputText {
            text: "what is this".into(),
        }])]);
        assert_eq!(first_video_input_item(&input), None);
    }

    #[test]
    fn responses_video_is_found_and_located() {
        let input = ResponseInput::Items(vec![
            simple_item(vec![ResponseContentPart::InputText {
                text: "first".into(),
            }]),
            simple_item(vec![ResponseContentPart::InputVideo {
                video_url: Some(serde_json::json!({"url": "https://example.invalid/a.mp4"})),
            }]),
        ]);
        assert_eq!(first_video_input_item(&input), Some(1));
    }

    #[test]
    fn responses_accepts_the_chat_spelling_too() {
        let input = ResponseInput::Items(vec![simple_item(vec![ResponseContentPart::VideoUrl {
            video_url: Some(serde_json::json!({"url": "https://example.invalid/a.mp4"})),
        }])]);
        assert_eq!(first_video_input_item(&input), Some(0));
    }

    #[test]
    fn responses_video_in_a_structured_message_counts() {
        let input = ResponseInput::Items(vec![ResponseInputOutputItem::Message {
            id: "msg_1".into(),
            role: "user".into(),
            content: vec![ResponseContentPart::InputVideo { video_url: None }],
            status: None,
        }]);
        assert_eq!(first_video_input_item(&input), Some(0));
    }

    /// An unrecognised part deserializes to `Unknown`, so it must not be
    /// mistaken for video; an image is still accepted.
    #[test]
    fn responses_unknown_part_is_not_treated_as_video() {
        let input = ResponseInput::Items(vec![simple_item(vec![ResponseContentPart::Unknown(
            serde_json::json!({"type": "input_image", "image_url": "https://example.invalid/a.png"}),
        )])]);
        assert_eq!(first_video_input_item(&input), None);
    }

    /// `input_video` must survive deserialization as its own variant rather
    /// than collapsing into the `Unknown` catch-all.
    #[test]
    fn responses_video_part_deserializes_to_its_own_variant() {
        let part: ResponseContentPart = serde_json::from_str(
            r#"{"type":"input_video","video_url":{"url":"https://example.invalid/a.mp4"}}"#,
        )
        .unwrap();
        assert!(matches!(part, ResponseContentPart::InputVideo { .. }));
    }

    // ---- image counting, chat -------------------------------------------

    fn image_part(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn no_images_in_plain_text() {
        assert_eq!(count_image_parts(&[text_user("hello")]), 0);
        assert_eq!(count_image_parts(&[]), 0);
    }

    #[test]
    fn no_images_in_text_and_video_parts() {
        let message = user_with(vec![
            ContentPart::Text {
                text: "describe".into(),
            },
            ContentPart::VideoUrl {
                video_url: VideoUrl {
                    url: "https://example.invalid/a.mp4".into(),
                    ..Default::default()
                },
            },
        ]);
        assert_eq!(count_image_parts(&[message]), 0);
    }

    #[test]
    fn one_image_is_counted() {
        let message = user_with(vec![
            ContentPart::Text {
                text: "what is this".into(),
            },
            image_part("https://example.invalid/a.png"),
        ]);
        assert_eq!(count_image_parts(&[message]), 1);
    }

    #[test]
    fn images_are_summed_across_messages() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("be brief".into()),
                name: None,
            },
            user_with(vec![
                image_part("https://example.invalid/a.png"),
                image_part("https://example.invalid/b.png"),
            ]),
            text_user("and this one?"),
            user_with(vec![
                ContentPart::Text {
                    text: "compare".into(),
                },
                image_part("https://example.invalid/c.png"),
            ]),
        ];
        assert_eq!(count_image_parts(&messages), 3);
    }

    #[test]
    fn images_in_an_assistant_turn_count() {
        let messages = vec![
            user_with(vec![image_part("https://example.invalid/a.png")]),
            ChatMessage::Assistant {
                content: Some(MessageContent::Parts(vec![image_part(
                    "https://example.invalid/b.png",
                )])),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            },
        ];
        assert_eq!(count_image_parts(&messages), 2);
    }

    #[test]
    fn assistant_turn_without_content_is_not_counted() {
        let message = ChatMessage::Assistant {
            content: None,
            name: None,
            tool_calls: None,
            reasoning_content: None,
        };
        assert_eq!(count_image_parts(&[message]), 0);
    }

    // ---- image counting, responses --------------------------------------

    fn responses_image(url: &str) -> ResponseContentPart {
        ResponseContentPart::Unknown(serde_json::json!({"type": "input_image", "image_url": url}))
    }

    #[test]
    fn responses_no_images_in_plain_string_input() {
        assert_eq!(
            count_image_input_parts(&ResponseInput::Text("hello".into())),
            0
        );
    }

    #[test]
    fn responses_no_images_in_text_video_or_other_unknown_parts() {
        let input = ResponseInput::Items(vec![simple_item(vec![
            ResponseContentPart::InputText {
                text: "what is this".into(),
            },
            ResponseContentPart::InputVideo { video_url: None },
            ResponseContentPart::Unknown(
                serde_json::json!({"type": "input_file", "file": {"id": "f_1"}}),
            ),
        ])]);
        assert_eq!(count_image_input_parts(&input), 0);
    }

    #[test]
    fn responses_one_image_is_counted() {
        let input = ResponseInput::Items(vec![simple_item(vec![
            ResponseContentPart::InputText {
                text: "what is this".into(),
            },
            responses_image("https://example.invalid/a.png"),
        ])]);
        assert_eq!(count_image_input_parts(&input), 1);
    }

    #[test]
    fn responses_images_are_summed_across_items() {
        let input = ResponseInput::Items(vec![
            simple_item(vec![
                responses_image("https://example.invalid/a.png"),
                responses_image("https://example.invalid/b.png"),
            ]),
            ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String("and this?".into()),
                role: "user".into(),
                r#type: None,
            },
            ResponseInputOutputItem::Message {
                id: "msg_1".into(),
                role: "assistant".into(),
                content: vec![responses_image("https://example.invalid/c.png")],
                status: None,
            },
        ]);
        assert_eq!(count_image_input_parts(&input), 3);
    }

    #[test]
    fn responses_accepts_the_chat_image_spelling_too() {
        let input = ResponseInput::Items(vec![simple_item(vec![ResponseContentPart::Unknown(
            serde_json::json!({"type": "image_url", "image_url": {"url": "https://example.invalid/a.png"}}),
        )])]);
        assert_eq!(count_image_input_parts(&input), 1);
    }

    /// Guards the assumption behind `is_image_input_part`: an `input_image`
    /// part currently lands in `Unknown`. If the protocol crate ever models
    /// it, this fails and the tag check must move to the new variant.
    #[test]
    fn responses_image_part_deserialized_from_json_is_counted() {
        let input: ResponseInput = serde_json::from_str(
            r#"[{"role":"user","content":[
                {"type":"input_text","text":"what is this"},
                {"type":"input_image","image_url":"https://example.invalid/a.png","detail":"high"}
            ]}]"#,
        )
        .unwrap();
        assert_eq!(count_image_input_parts(&input), 1);
    }
}
