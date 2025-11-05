#!/usr/bin/env python3
"""
Alibaba Cloud Model Studio (OpenAI-compatible) video URL streaming test.

Demonstrates Qwen Omni/VL video+text input using streaming Chat Completions.
Defaults match the docs example; customize with CLI flags.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


def load_dotenv_if_present(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def main():
    import argparse
    try:
        from openai import OpenAI
    except Exception:
        print("❌ Missing dependency: openai. Install with: pip install -U openai")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Alibaba Model Studio (OpenAI-compatible) video streaming test")
    parser.add_argument("--video-url", type=str, default="https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241115/cqqkru/1.mp4")
    parser.add_argument("--prompt", type=str, default="What is the content of the video?")
    parser.add_argument("--model", type=str, default="qwen3-omni-flash", help="e.g., qwen3-omni-flash, qwen-omni-turbo")
    parser.add_argument("--api-base", type=str, default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", type=str, default="DASHSCOPE_API_KEY")
    parser.add_argument("--modalities", type=str, default="text", help="Comma-separated: text or text,audio")
    parser.add_argument("--voice", type=str, default="Cherry", help="Audio voice, when modalities include audio")
    parser.add_argument("--audio-format", type=str, default="wav", help="Audio format, when modalities include audio")
    parser.add_argument("--dotenv", type=str, default=".env", help="Path to .env (optional)")
    args = parser.parse_args()

    if args.dotenv:
        load_dotenv_if_present(args.dotenv)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"❌ API key not found in env var {args.api_key_env}")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    modalities: List[str] = [m.strip() for m in args.modalities.split(',') if m.strip()]
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": args.video_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if modalities:
        payload["modalities"] = modalities
        if "audio" in modalities:
            payload["audio"] = {"voice": args.voice, "format": args.audio_format}

    print("=" * 70)
    print("🎥 Alibaba Model Studio video streaming test")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Video: {args.video_url}")
    print(f"Modalities: {modalities}")
    print()

    try:
        stream = client.chat.completions.create(**payload)
        for chunk in stream:
            # Print text deltas inline; usage arrives as a separate event
            try:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    print(delta)
                else:
                    print(chunk.usage)
            except Exception:
                try:
                    print(chunk)
                except Exception:
                    pass
    except Exception as e:
        print(f"❌ Error: {e}")
        print("For more info: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
        sys.exit(2)


if __name__ == "__main__":
    main()

