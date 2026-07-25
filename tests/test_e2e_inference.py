import time
import requests
import json
import uuid
import threading
from pravaha import pravaha_core

def test_e2e_inference_server():
    """
    Tests the full lifecycle:
    1. Instantiate the Rust TokenBridge in Python.
    2. Spawn the Rust Axum HTTP server in a background thread via PyO3.
    3. Make a client HTTP POST request to /v1/completions.
    4. Simulate the Python Engine generating tokens and pushing to the Bridge.
    5. Consume the SSE stream and assert correctness and Request-ID matching.
    """
    # 1. Setup Bridge & Start Server
    bridge = pravaha_core.TokenBridge()
    port = 8443
    
    # Start the server in the background (Tokio thread)
    pravaha_core.start_server_bg(bridge, port)
    
    # Wait for Tokio TcpListener to bind
    time.sleep(0.5)
    
    # 2. Test /health endpoint
    health_url = f"http://127.0.0.1:{port}/health"
    response = requests.get(health_url)
    assert response.status_code == 200, "Health endpoint failed"
    health_data = response.json()
    assert health_data["status"] == "ok"
    assert "uptime_ms" in health_data
    assert health_data["uptime_ms"] >= 0

    # 3. Test /v1/completions SSE Stream
    completions_url = f"http://127.0.0.1:{port}/v1/completions"
    payload = {
        "prompt": "Hello, how are you?",
        "max_tokens": 5,
        "temperature": 0.7,
        "stream": True
    }
    
    # Send request in a separate thread because we need to push tokens from Python simultaneously!
    received_tokens = []
    received_request_id = None
    header_request_id = None

    def client_request():
        nonlocal header_request_id
        # We need stream=True to read SSE chunk by chunk
        with requests.post(completions_url, json=payload, stream=True, timeout=5) as r:
            header_request_id = r.headers.get("X-Request-ID")
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data = decoded_line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                received_tokens.append(chunk["choices"][0]["text"])
                                nonlocal received_request_id
                                received_request_id = chunk["id"]
                        except json.JSONDecodeError:
                            pass

    client_thread = threading.Thread(target=client_request)
    client_thread.start()

    # The Axum server receives the prompt, creates a request ID, registers a channel in TokenBridge,
    # and waits for tokens. We don't know the exact Request ID generated yet, but we can peek at active requests!
    # Wait a tiny bit for the request to reach the server.
    time.sleep(0.5)
    
    # Get active streams from the bridge
    active_requests = bridge.get_active_streams()
    assert len(active_requests) == 1, f"Expected 1 active stream, got {len(active_requests)}"
    
    req_id = active_requests[0]
    
    # 4. Simulate Python Inference Engine pushing tokens
    test_tokens = ["Hello", " ", "World", "!", " [DONE]"]
    for token in test_tokens:
        bridge.send_token(req_id, token)
        time.sleep(0.01)

    # Clean up the stream
    bridge.finish_stream(req_id)
    
    client_thread.join(timeout=2)
    
    # 5. Assertions
    assert len(received_tokens) == 5, f"Expected 5 tokens, got {len(received_tokens)}"
    assert "".join(received_tokens) == "Hello World! [DONE]"
    
    # Validate the Request ID matches between the HTTP Header and the SSE payload chunk
    assert header_request_id is not None, "Missing X-Request-ID in HTTP headers"
    assert received_request_id is not None, "Missing Request ID in SSE JSON chunk"
    assert header_request_id == received_request_id, "Request ID mismatch between Header and Payload!"
    assert req_id == header_request_id, "Bridge stream ID did not match Request ID"

    print("E2E Integration Test Passed!")
