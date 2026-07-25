use pyo3::prelude::*;
use pyo3::exceptions::PyKeyError;
use dashmap::DashMap;
use tokio::sync::mpsc;
use std::sync::{Arc, Mutex as StdMutex};
use std::sync::mpsc as std_mpsc;

/// A PyO3 class to bridge token streams from Python to Rust.
#[pyclass]
#[derive(Clone)]
pub struct TokenBridge {
    pub(crate) senders: Arc<DashMap<String, mpsc::Sender<String>>>,
    pub(crate) requests_tx: std_mpsc::Sender<(String, String)>,
    pub(crate) requests_rx: Arc<StdMutex<std_mpsc::Receiver<(String, String)>>>,
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
        let (tx, rx) = std_mpsc::channel();
        Self {
            senders: Arc::new(DashMap::new()),
            requests_tx: tx,
            requests_rx: Arc::new(StdMutex::new(rx)),
        }
    }

    /// Creates a stream for the given request ID, returning nothing to Python.
    pub fn create_stream(&self, _request_id: String) -> PyResult<()> {
        Ok(())
    }

    /// Send a token for a given request ID.
    pub fn send_token(&self, request_id: String, token: String) -> PyResult<()> {
        if let Some(sender) = self.senders.get(&request_id) {
            // Use blocking_send to avoid dropping tokens when the channel is full.
            if let Err(_e) = sender.blocking_send(token) {
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

    /// Return all active request IDs currently streaming.
    pub fn get_active_streams(&self) -> Vec<String> {
        let mut keys = Vec::new();
        for item in self.senders.iter() {
            keys.push(item.key().clone());
        }
        keys
    }

    /// Poll for a new generation request from the Rust server.
    /// Returns (request_id, prompt) if available, otherwise None.
    pub fn poll_request(&self) -> Option<(String, String)> {
        if let Ok(rx) = self.requests_rx.lock() {
            rx.try_recv().ok()
        } else {
            None
        }
    }
}

impl TokenBridge {
    /// Internal Rust method to create a stream and return the receiver.
    pub fn register_stream(&self, request_id: String, buffer_size: usize) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel(buffer_size);
        self.senders.insert(request_id, tx);
        rx
    }

    /// Internal Rust method to push a request to Python.
    pub fn push_request(&self, request_id: String, prompt: String) {
        let _ = self.requests_tx.send((request_id, prompt));
    }
}
