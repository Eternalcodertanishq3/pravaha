"""Pravāha Inference Server — One-Command Launcher.

Usage:
    python serve.py                          # Default: gpt2 on port 8000
    python serve.py --port 9000              # Custom port
    python serve.py --host 127.0.0.1         # Localhost only
"""

import sys
from pathlib import Path

# Ensure pravaha is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="🌊 Pravāha Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve.py                    Start server on port 8000
  python serve.py --port 9000       Use custom port
  python serve.py --reload          Auto-reload on code changes (dev mode)

Once running, visit:
  http://localhost:8000/docs         Interactive API documentation
  http://localhost:8000/v1/models    List loaded models
  http://localhost:8000/metrics      Live GPU telemetry
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🌊 Pravāha Inference Server")
    print(f"  🔗 http://localhost:{args.port}")
    print(f"  📄 http://localhost:{args.port}/docs")
    print(f"  📊 http://localhost:{args.port}/metrics")
    print("=" * 60)
    
    uvicorn.run(
        "pravaha.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
