from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simplellm.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the educational SimpleLLM API.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.getenv("SIMPLELLM_CHECKPOINT"),
        help="Path to a .pt checkpoint. Can also be set with SIMPLELLM_CHECKPOINT.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None, help="Device such as cpu, cuda, or cuda:0.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Hugging Face tokenizer name.")
    parser.add_argument("--no-strict", action="store_true", help="Load checkpoint weights with strict=False.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        checkpoint_path=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
        strict=not args.no_strict,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
