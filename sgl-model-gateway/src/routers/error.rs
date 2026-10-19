use axum::{
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'a str,
    message: &'a str,
}

pub const HEADER_X_SMG_ERROR_CODE: &str = "X-SMG-Error-Code";

pub fn internal_error(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, code, message)
}

pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, code, message)
}

pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, code, message)
}

pub fn service_unavailable(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::SERVICE_UNAVAILABLE, code, message)
}

pub fn failed_dependency(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::FAILED_DEPENDENCY, code, message)
}

pub fn not_implemented(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_IMPLEMENTED, code, message)
}

pub fn bad_gateway(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_GATEWAY, code, message)
}

pub fn method_not_allowed(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::METHOD_NOT_ALLOWED, code, message)
}

/// Shed load with a retry hint.
///
/// `Retry-After` is what turns a rejection into backpressure: without it a
/// client that retries immediately keeps the gateway at its ceiling, and the
/// rejections do not shed any load.
pub fn too_many_requests(
    code: impl Into<String>,
    message: impl Into<String>,
    retry_after_secs: u64,
) -> Response {
    let mut response = create_error(StatusCode::TOO_MANY_REQUESTS, code, message);
    if let Ok(value) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        response.headers_mut().insert(header::RETRY_AFTER, value);
    }
    response
}

pub fn create_error(
    status: StatusCode,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    let code_str = code.into();
    let message_str = message.into();

    let mut headers = HeaderMap::with_capacity(1);
    headers.insert(
        HEADER_X_SMG_ERROR_CODE,
        HeaderValue::from_str(&code_str).unwrap(),
    );

    (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: status_code_to_str(status),
                code: &code_str,
                message: &message_str,
            },
        }),
    )
        .into_response()
}

fn status_code_to_str(status_code: StatusCode) -> &'static str {
    status_code
        .canonical_reason()
        .unwrap_or("Unknown Status Code")
}

/// Best-effort `error_code` label for a non-2xx a worker produced rather than
/// us. Upstream failures are forwarded verbatim and carry no
/// `X-SMG-Error-Code`, so without this the whole class records as empty.
pub fn upstream_error_code(body: &[u8], status: StatusCode) -> String {
    #[derive(serde::Deserialize)]
    struct Body {
        error: Option<Detail>,
        // Some engines answer flat rather than nested.
        code: Option<serde_json::Value>,
        #[serde(rename = "type")]
        kind: Option<String>,
    }
    #[derive(serde::Deserialize)]
    struct Detail {
        code: Option<serde_json::Value>,
        #[serde(rename = "type")]
        kind: Option<String>,
    }

    fn as_code(v: &serde_json::Value) -> Option<String> {
        match v {
            serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    }

    let parsed: Option<Body> = serde_json::from_slice(body).ok();
    let from_body = parsed.and_then(|b| {
        let (code, kind) = match b.error {
            Some(d) => (d.code, d.kind),
            None => (b.code, b.kind),
        };
        code.as_ref()
            .and_then(as_code)
            .or_else(|| kind.filter(|k| !k.is_empty()))
    });

    // Sanitise to a short slug: an engine may put a whole sentence in `code`,
    // and each distinct value becomes its own Prometheus time series.
    match from_body {
        Some(raw) => {
            let slug: String = raw
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .take(48)
                .collect();
            format!("upstream_{}", slug.trim_matches('_').to_ascii_lowercase())
        }
        None => format!("upstream_http_{}", status.as_u16()),
    }
}

pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(HEADER_X_SMG_ERROR_CODE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}
