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
  http://localhost:8000/world        3D Swarm Visualizer (NEW!)
  http://localhost:8000/v1/models    List loaded models
  http://localhost:8000/metrics      Live GPU telemetry
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--tui", action="store_true", help="Launch the TUI Dashboard")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file")
    parser.add_argument("--ssl-certfile", default=None, help="SSL certificate file")
    
    args = parser.parse_args()
    
    scheme = "https" if args.ssl_keyfile and args.ssl_certfile else "http"
    
    print("=" * 60)
    print("  🌊 Pravāha Inference Server")
    print(f"  🔗 {scheme}://localhost:{args.port}")
    print(f"  📄 {scheme}://localhost:{args.port}/docs")
    print(f"  📊 {scheme}://localhost:{args.port}/metrics")
    print(f"  🌐 {scheme}://localhost:{args.port}/world  (3D Swarm Visualizer)")
    print("=" * 60)
    
    if args.tui:
        import asyncio
        import uvicorn
        from pravaha.tui.app import PravahaTUI, get_connector
        from pravaha.serving.app import create_app
        
        print("\n[+] Launching TUI Dashboard alongside backend...\n")
        app = create_app()
        config = uvicorn.Config(
            app, 
            host=args.host, 
            port=args.port, 
            log_level="error",
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
        )
        server = uvicorn.Server(config)
        
        tui = PravahaTUI()
        
        async def run_both():
            server_task = asyncio.create_task(server.serve())
            
            # Wait for lifespan to initialize engine
            for _ in range(20):
                if hasattr(app.state, "engine") and app.state.engine:
                    break
                await asyncio.sleep(0.5)
                
            if hasattr(app.state, "engine") and app.state.engine:
                conn = get_connector()
                conn.attach_engine(app.state.engine)
                
            await tui.run_async()
            server.should_exit = True
            await server_task

        try:
            asyncio.run(run_both())
        except KeyboardInterrupt:
            pass
    else:
        uvicorn.run(
            "pravaha.serving.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
        )


if __name__ == "__main__":
    main()
