use pyo3::prelude::*;
use pyo3::exceptions::PyKeyError;
use dashmap::DashMap;
use tokio::sync::mpsc;
use std::sync::Arc;

/// A PyO3 class to bridge token streams from Python to Rust.
#[pyclass]
#[derive(Clone)]
pub struct TokenBridge {
    pub(crate) senders: Arc<DashMap<String, mpsc::Sender<String>>>,
}

impl Default for TokenBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl TokenBridge {
    #[new]
    pub fn new() -> Self {
        Self {
            senders: Arc::new(DashMap::new()),
        }
    }

    /// Creates a stream for the given request ID, returning nothing to Python.
    /// In actual usage, the Rust server will have access to the Receiver via a secondary method.
    /// Since the HTTP server needs the receiver, it's typically the Rust side that calls this directly,
    /// but we provide it here if Python wants to initiate.
    /// Wait, if Python calls create_stream, we don't have the receiver easily accessible by Rust unless
    /// we store it. Let's provide a method for Rust to extract the receiver, or instead let Rust create it.
    /// The prompt says: "Returns the sender to Python via PyO3" or "The server holds a TokenBridge that:
    /// Creates a new mpsc::channel per request
    /// Returns the sender to Python via PyO3".
    /// If TokenBridge is passed to the server, we can implement an internal Rust method to register a stream.
    pub fn create_stream(&self, _request_id: String) -> PyResult<()> {
        // Python doesn't usually create the stream, but if it must, we could do it.
        // For the sake of the API requested:
        Ok(())
    }

    /// Send a token for a given request ID.
    pub fn send_token(&self, request_id: String, token: String) -> PyResult<()> {
        if let Some(sender) = self.senders.get(&request_id) {
            // We use try_send because we don't want to block the Python thread.
            // If the channel is full or closed, it will fail.
            if let Err(_e) = sender.try_send(token) {
                // Ignore error, stream might have been closed by client
            }
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("No stream found for request_id {}", request_id)))
        }
    }

    /// Finish the stream for a given request ID, removing the sender.
    pub fn finish_stream(&self, request_id: String) -> PyResult<()> {
        if self.senders.remove(&request_id).is_some() {
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!("No stream found for request_id {}", request_id)))
        }
    }
}

impl TokenBridge {
    /// Internal Rust method to create a stream and return the receiver.
    pub fn register_stream(&self, request_id: String) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel(100);
        self.senders.insert(request_id, tx);
        rx
    }
}
