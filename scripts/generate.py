from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simplellm import SimpleLLMInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the educational SimpleLLM model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None, help="Device such as cpu, cuda, or cuda:0.")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Hugging Face tokenizer name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = SimpleLLMInference(
        checkpoint_path=args.checkpoint,
        tokenizer_name=args.tokenizer,
        device=args.device,
    )
    output = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(output)


if __name__ == "__main__":
    main()
