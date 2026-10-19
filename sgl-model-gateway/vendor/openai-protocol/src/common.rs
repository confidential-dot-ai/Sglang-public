use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator;

use super::UNKNOWN_MODEL_ID;

// ============================================================================
// Default value helpers
// ============================================================================

/// Default model value when not specified
pub(crate) fn default_model() -> String {
    UNKNOWN_MODEL_ID.to_string()
}

/// Helper function for serde default value (returns true)
pub fn default_true() -> bool {
    true
}

// ============================================================================
// GenerationRequest Trait
// ============================================================================

/// Trait for unified access to generation request properties
/// Implemented by ChatCompletionRequest, CompletionRequest, GenerateRequest,
/// EmbeddingRequest, RerankRequest, and ResponsesRequest
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;

    /// Extract text content for `prefix_hash` routing specifically: like
    /// `extract_text_for_routing`, but with any leading system/developer
    /// content excluded.
    ///
    /// `prefix_hash` hashes only the first `prefix_token_count` tokens of
    /// whatever text it's given (see `PrefixHashPolicy`) to pick a worker,
    /// aiming for one worker per *distinct task*. A leading system/developer
    /// message shared verbatim across many otherwise-unrelated requests (a
    /// common agent-harness pattern: the same boilerplate instructions
    /// prepended to every task) can be longer than that window, in which
    /// case every such request hashes identically and collapses onto one
    /// worker regardless of how many workers exist -- confirmed live against
    /// a real SWE-agent trace dataset where 49/50 sampled conversations
    /// shared an identical leading 263 tokens (2026-07-24).
    ///
    /// `extract_text_for_routing` itself must keep including that content --
    /// `cache_aware`'s radix-tree matching needs the FULL served text
    /// (system prompt included, since that's what's actually resident in
    /// worker KV caches) to route correctly. This is a separate method
    /// specifically so `prefix_hash` can route on the part of the request
    /// that actually distinguishes one task from another, without changing
    /// what `cache_aware` sees.
    ///
    /// Default: identical to `extract_text_for_routing` (no system/developer
    /// content to separate out for most request types -- only chat requests
    /// have that structure).
    fn extract_text_for_prefix_hash_routing(&self) -> String {
        self.extract_text_for_routing()
    }
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    /// Get the number of items in the StringOrArray
    pub fn len(&self) -> usize {
        match self {
            StringOrArray::String(_) => 1,
            StringOrArray::Array(arr) => arr.len(),
        }
    }

    /// Check if the StringOrArray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrArray::String(s) => s.is_empty(),
            StringOrArray::Array(arr) => arr.is_empty(),
        }
    }

    /// Convert to a vector of strings (clones the data)
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s.clone()],
            StringOrArray::Array(arr) => arr.clone(),
        }
    }

    /// Returns an iterator over string references without cloning.
    /// Use this instead of `to_vec()` when you only need to iterate.
    pub fn iter(&self) -> StringOrArrayIter<'_> {
        StringOrArrayIter {
            inner: self,
            index: 0,
        }
    }

    /// Returns the first string, or None if empty
    pub fn first(&self) -> Option<&str> {
        match self {
            StringOrArray::String(s) => {
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            }
            StringOrArray::Array(arr) => arr.first().map(|s| s.as_str()),
        }
    }
}

/// Iterator over StringOrArray that yields string references without cloning
pub struct StringOrArrayIter<'a> {
    inner: &'a StringOrArray,
    index: usize,
}

impl<'a> Iterator for StringOrArrayIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            StringOrArray::String(s) => {
                if self.index == 0 {
                    self.index = 1;
                    Some(s.as_str())
                } else {
                    None
                }
            }
            StringOrArray::Array(arr) => {
                if self.index < arr.len() {
                    let item = &arr[self.index];
                    self.index += 1;
                    Some(item.as_str())
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = match self.inner {
            StringOrArray::String(_) => 1 - self.index,
            StringOrArray::Array(arr) => arr.len() - self.index,
        };
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for StringOrArrayIter<'a> {}

/// Validates stop sequences (max 4, non-empty strings)
/// Used by both ChatCompletionRequest and ResponsesRequest
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    match stop {
        StringOrArray::String(s) => {
            if s.is_empty() {
                return Err(validator::ValidationError::new(
                    "stop sequences cannot be empty",
                ));
            }
        }
        StringOrArray::Array(arr) => {
            if arr.len() > 4 {
                return Err(validator::ValidationError::new(
                    "maximum 4 stop sequences allowed",
                ));
            }
            for s in arr {
                if s.is_empty() {
                    return Err(validator::ValidationError::new(
                        "stop sequences cannot be empty",
                    ));
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "default", "low", or "high"
    /// Every other key the caller sent, preserved verbatim.
    ///
    /// The router deserializes a request and serializes it again on the way to
    /// a worker, so any field this struct does not name is silently dropped in
    /// between -- the client sees its option accepted and the worker never
    /// learns it existed. That is not hypothetical: `fps` and
    /// `max_long_side_pixel` were both validated correctly by the worker and
    /// both reached it as `None`, so out-of-range values were served and
    /// resolution tiers had no effect. Capturing the remainder keeps this
    /// layer from having to know every model's media options, and stops the
    /// next one from failing the same silent way.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct VideoUrl {
    pub url: String,
    /// See `ImageUrl::extra`. Video carried nothing but a URL, so `detail`,
    /// `fps` and the resolution caps were all dropped here.
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, Value>,
}

// ============================================================================
// Response Format (for structured outputs)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

/// Tool choice value for simple string options
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for both Chat Completion and Responses APIs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String, // "allowed_tools"
        mode: String, // "auto" | "required" TODO: need validation
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

impl ToolChoice {
    /// Serialize tool_choice to string for ResponsesResponse
    ///
    /// Returns the JSON-serialized tool_choice or "auto" as default
    pub fn serialize_to_string(tool_choice: &Option<ToolChoice>) -> String {
        tool_choice
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for ToolChoice::AllowedTools
///
/// Represents a reference to a specific tool in the allowed_tools array.
/// Different tool types have different required fields.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    /// Reference to a function tool
    #[serde(rename = "function")]
    Function { name: String },

    /// Reference to an MCP tool
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// File search hosted tool
    #[serde(rename = "file_search")]
    FileSearch,

    /// Web search preview hosted tool
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,

    /// Computer use preview hosted tool
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,

    /// Code interpreter hosted tool
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,

    /// Image generation hosted tool
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{}", name),
            ToolReference::Mcp { server_label, name } => {
                if let Some(n) = name {
                    format!("mcp:{}:{}", server_label, n)
                } else {
                    format!("mcp:{}", server_label)
                }
            }
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }

    /// Get the tool name if this is a function tool
    pub fn function_name(&self) -> Option<&str> {
        match self {
            ToolReference::Function { name } => Some(name.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value, // JSON Schema
    /// Whether to enable strict schema adherence (OpenAI structured outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum FunctionCall {
    None,
    Auto,
    Function { name: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>, // JSON string
}

// ============================================================================
// Usage and Logging
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

/// Usage information (used by rerank and other endpoints)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatLogProbs {
    Detailed {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ChatLogProbsContent>>,
    },
    Raw(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

/// LoRA adapter path - can be single path or batch of paths (SGLang extension)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LoRAPath {
    Single(Option<String>),
    Batch(Vec<Option<String>>),
}

#[cfg(test)]
mod media_passthrough_tests {
    use super::*;

    /// The router deserializes a request and serializes it again on the way to
    /// a worker. Any media option this layer does not preserve is dropped in
    /// between, so the client sees it accepted and the worker never learns it
    /// existed -- silently, with a 200.
    fn round_trip(json: &str) -> serde_json::Value {
        let part: ContentPart = serde_json::from_str(json).expect("deserialize");
        serde_json::to_value(&part).expect("serialize")
    }

    #[test]
    fn video_url_preserves_fps_and_resolution_caps() {
        let out = round_trip(
            r#"{"type":"video_url","video_url":{"url":"u","fps":2.0,"max_long_side_pixel":1008,"detail":"default"}}"#,
        );
        let v = &out["video_url"];
        assert_eq!(v["url"], "u");
        assert_eq!(v["fps"], 2.0);
        assert_eq!(v["max_long_side_pixel"], 1008);
        assert_eq!(v["detail"], "default");
    }

    #[test]
    fn image_url_preserves_resolution_caps() {
        let out = round_trip(
            r#"{"type":"image_url","image_url":{"url":"u","detail":"default","max_long_side_pixel":504,"max_total_pixels":1000}}"#,
        );
        let v = &out["image_url"];
        assert_eq!(v["detail"], "default");
        assert_eq!(v["max_long_side_pixel"], 504);
        assert_eq!(v["max_total_pixels"], 1000);
    }

    #[test]
    fn an_out_of_range_value_survives_to_reach_worker_validation() {
        // The router is not the validator: it must hand fps=100 to the worker
        // intact so the worker can reject it, rather than dropping it and
        // turning a 400 into a 200.
        let out = round_trip(r#"{"type":"video_url","video_url":{"url":"u","fps":100}}"#);
        assert_eq!(out["video_url"]["fps"], 100);
    }

    #[test]
    fn plain_urls_gain_no_extra_keys() {
        let out = round_trip(r#"{"type":"video_url","video_url":{"url":"u"}}"#);
        let v = out["video_url"].as_object().unwrap();
        assert_eq!(v.len(), 1, "unexpected keys: {:?}", v.keys().collect::<Vec<_>>());
    }
}
