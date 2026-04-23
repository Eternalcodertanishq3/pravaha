# Deployment Guide

## Docker

### CPU

```bash
docker build -f docker/Dockerfile -t pravaha:latest .
docker run -p 8000:8000 pravaha:latest
```

### GPU (CUDA)

```bash
docker build -f docker/Dockerfile.cuda -t pravaha:cuda .
docker run --gpus all -p 8000:8000 pravaha:cuda
```

### Full Stack (Pravaha + Prometheus + Grafana)

```bash
docker compose -f docker/docker-compose.yml up
```

- Pravaha: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/pravaha)

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pravaha
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pravaha
  template:
    metadata:
      labels:
        app: pravaha
    spec:
      containers:
        - name: pravaha
          image: pravaha:cuda
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 1
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 30
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | gpt2 | Model to serve |
| `QUANTIZE` | none | Quantization mode |
| `SWARM` | false | Enable swarm |
| `RAG` | false | Enable RAG |
| `PORT` | 8000 | Server port |
