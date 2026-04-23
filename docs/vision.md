# Vision Routing

Pravāha supports multimodal inference by routing vision requests to appropriate models (e.g., LLaVA).

## Usage

```bash
curl -X POST http://localhost:8000/v1/vision/complete \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-1.5-7b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'
```

## Architecture

1. **Detector** — identifies image format and validates input
2. **Preprocessor** — resizes and normalizes images for the vision encoder
3. **VisionEngine** — routes to the appropriate multimodal model
