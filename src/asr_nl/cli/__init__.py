"""CLI entry point for the Gradio app."""

import argparse
from ..gradio import launch

def main():
    p = argparse.ArgumentParser(description="Launch Dutch ASR Leaderboard Gradio app")
    p.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    p.add_argument("--port", type=int, default=7860, help="Port to bind to")
    args = p.parse_args()
    launch(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
