#!/usr/bin/env python3
"""
Alibaba Cloud Model Studio (OpenAI-compatible) smoke test.

Loads API key from the environment or a local .env (if present) and
performs a simple Chat Completions call against a Qwen model.

Example:
  cr-ali-test --model qwen-plus \
    --api-base https://dashscope-intl.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_dotenv_if_present(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def main():
    import argparse
    try:
        from openai import OpenAI
    except Exception as e:
        print("❌ Missing dependency: openai. Install with: pip install -U openai")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Alibaba Model Studio (OpenAI-compatible) smoke test")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Model name, e.g., qwen-plus, qwen-vl-max")
    parser.add_argument("--api-base", type=str, default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", help="Base URL (intl or cn)")
    parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY", help="Env var containing API key")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    parser.add_argument("--question", type=str, default="Who are you?")
    parser.add_argument("--dotenv", type=str, default=".env", help="Optional path to .env to load")
    args = parser.parse_args()

    # Load .env if present
    if args.dotenv:
        load_dotenv_if_present(args.dotenv)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"❌ API key not found in env var {args.api_key_env}. Add it to your shell or .env file.")
        sys.exit(1)

    try:
        client = OpenAI(api_key=api_key, base_url=args.api_base)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.question},
            ],
        )
        # Print the first choice content if present
        content = None
        try:
            content = completion.choices[0].message.content
        except Exception:
            content = str(completion)
        print(content)
    except Exception as e:
        print(f"Error message: {e}")
        print("For more info: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
        sys.exit(2)


if __name__ == "__main__":
    main()

