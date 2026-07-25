use axum::{
    extract::{State, Json, Extension},
    response::sse::{Event, Sse},
    routing::{get, post},
    Router,
};
use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::trace::TraceLayer;
use uuid::Uuid;
use axum::http::{Request, HeaderMap};
use axum::middleware::Next;
use axum::response::Response;
use tokio::signal;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::token_bridge::TokenBridge;

#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Clone)]
pub struct AppState {
    pub bridge: TokenBridge,
    pub start_time: Instant,
}

#[derive(Clone, Debug)]
pub struct RequestId(pub String);

async fn request_id_middleware(mut req: Request<axum::body::Body>, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    req.extensions_mut().insert(RequestId(request_id.clone()));
    
    let mut response = next.run(req).await;
    if let Ok(val) = request_id.parse() {
        response.headers_mut().insert("X-Request-ID", val);
    }
    response
}

pub async fn completions_handler(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    Json(payload): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // We now process the payload (e.g. log or send to Python engine queue)
    println!("Received prompt (ReqID: {}): {}", request_id.0, payload.prompt);
    
    // Register the stream with the exact request ID from the middleware
    let rx = state.bridge.register_stream(request_id.0.clone());

    let stream = ReceiverStream::new(rx).map(move |token| {
        let chunk = CompletionChunk {
            id: request_id.0.clone(),
            object: "text_completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            choices: vec![ChunkChoice {
                text: token,
                index: 0,
                finish_reason: None,
            }],
        };
        Ok::<Event, Infallible>(Event::default().json_data(chunk).unwrap_or_else(|_| Event::default()))
    });

    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::new())
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_ms: u128,
}

pub async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_ms: state.start_time.elapsed().as_millis(),
    })
}

pub struct RustTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl RustTokenizer {
    pub fn new(path: &str) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| format!("Encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer.decode(ids, false)
            .map_err(|e| format!("Decode error: {}", e))
    }
}

pub async fn run_server_async(bridge: TokenBridge, port: u16) {
    let state = AppState { 
        bridge,
        start_time: Instant::now(),
    };

    let app = Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/health", get(health_handler))
        .layer(axum::middleware::from_fn(request_id_middleware))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    if let Ok(listener) = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
        println!("Rust server listening on port {}", port);
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await;
    }
}

// Background thread launcher for Python
#[pyfunction]
pub fn start_server_bg(bridge: TokenBridge, port: u16) -> PyResult<()> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            run_server_async(bridge, port).await;
        });
    });
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate()) {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
