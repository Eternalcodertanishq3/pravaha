use axum::{
    extract::{State, Json},
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
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;
use tokio::signal;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::token_bridge::TokenBridge;

#[derive(Deserialize)]
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
}

// Generates Request ID
async fn request_id_middleware(mut req: Request<axum::body::Body>, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    req.headers_mut().insert(
        "X-Request-ID",
        request_id.parse().unwrap(),
    );
    let mut response = next.run(req).await;
    response.headers_mut().insert(
        "X-Request-ID",
        request_id.parse().unwrap(),
    );
    response
}

pub async fn completions_handler(
    State(state): State<AppState>,
    Json(payload): Json<CompletionRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let request_id = Uuid::new_v4().to_string();
    let rx = state.bridge.register_stream(request_id.clone());

    let stream = ReceiverStream::new(rx).map(move |token| {
        let chunk = CompletionChunk {
            id: request_id.clone(),
            object: "text_completion".to_string(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
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
    pub uptime_ms: u64,
}

pub async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_ms: 1000,
    })
}

pub struct RustTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl RustTokenizer {
    pub fn new(path: &str) -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_file(path).unwrap(),
        }
    }
    
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, false).unwrap().get_ids().to_vec()
    }
    
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, false).unwrap()
    }
}

pub async fn run_server(bridge: TokenBridge, port: u16) {
    let state = AppState { bridge };

    let app = Router::new()
        .route("/v1/completions", post(completions_handler))
        .route("/health", get(health_handler))
        .layer(axum::middleware::from_fn(request_id_middleware))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
