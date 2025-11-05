#!/usr/bin/env python3
"""
External Video Analyzer (scaffold)

CLI to analyze a video using external vision APIs. Designed to be dependency-light
and safe for large files: uses ffprobe/ffmpeg via subprocess (if available) and
urllib from the stdlib for HTTP. Providers are pluggable via a simple registry.

This is a scaffold: fill in provider-specific API calls once model docs/keys are provided.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import shutil
import random
import string
import mimetypes
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib import request, error


def ffprobe_metadata(video_path: Path) -> Dict[str, Any]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate:format=duration,bit_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return {"error": result.stderr.strip()}
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        # Normalize
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        def _fps(fr):
            if isinstance(fr, str) and "/" in fr:
                a, b = fr.split("/", 1)
                try:
                    a = float(a)
                    b = float(b)
                    return a / b if b else None
                except Exception:
                    return None
            try:
                return float(fr)
            except Exception:
                return None

        meta = {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "codec": stream.get("codec_name"),
            "fps": _fps(stream.get("avg_frame_rate")),
            "duration": _to_float(fmt.get("duration")),
            "bitrate": fmt.get("bit_rate"),
        }
        return {k: v for k, v in meta.items() if v is not None}
    except FileNotFoundError:
        return {"warning": "ffprobe not found"}
    except Exception as e:
        return {"error": str(e)}


def extract_frames(video_path: Path, out_dir: Path, num_frames: int, duration: Optional[float]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Choose an fps that yields ~num_frames across the duration; fallback to 1 fps.
    fps = None
    if duration and duration > 0:
        fps = max(0.0001, num_frames / duration)
    try:
        pattern = out_dir / "frame_%04d.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
        ]
        if fps:
            cmd += ["-vf", f"fps={fps}"]
        cmd += ["-frames:v", str(num_frames), str(pattern)]
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        # ffmpeg missing; nothing extracted
        return []
    except subprocess.CalledProcessError:
        # Best-effort: ignore extraction failure
        pass
    frames = sorted(out_dir.glob("frame_*.jpg"))
    # If more than requested, trim
    if len(frames) > num_frames:
        frames = frames[:num_frames]
    return frames


def _rand_boundary() -> str:
    return "----WebKitFormBoundary" + "".join(random.choices(string.ascii_letters + string.digits, k=16))


def http_multipart_post(url: str, fields: Dict[str, str], files: List[Tuple[str, Path]], headers: Dict[str, str], timeout: int = 120) -> Tuple[int, bytes]:
    boundary = _rand_boundary()
    body_iter = []
    # Text fields
    for name, value in fields.items():
        body_iter.append(f"--{boundary}\r\n".encode())
        body_iter.append(f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode())
        body_iter.append(value.encode())
        body_iter.append(b"\r\n")
    # File fields
    for form_name, path in files:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        body_iter.append(f"--{boundary}\r\n".encode())
        body_iter.append(
            f"Content-Disposition: form-data; name=\"{form_name}\"; filename=\"{path.name}\"\r\n".encode()
        )
        body_iter.append(f"Content-Type: {mime}\r\n\r\n".encode())
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                body_iter.append(chunk)
        body_iter.append(b"\r\n")
    body_iter.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_iter)
    req = request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


# Provider registry ---------------------------------------------------------

def analyze_noop(video_path: Path, frames: List[Path], prompt: str, **kwargs) -> Dict[str, Any]:
    return {
        "provider": "noop",
        "summary": "No-op analysis stub. Replace with a real provider.",
        "frame_count": len(frames),
        "prompt_used": prompt,
    }


def analyze_http(video_path: Path, frames: List[Path], prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Generic HTTP multipart POST.
    kwargs accepts:
      - api_base: str (endpoint URL)
      - api_key: Optional[str]
      - timeout: int
    Expects JSON response.
    """
    api_base: str = kwargs.get("api_base") or ""
    if not api_base:
        raise ValueError("api_base is required for provider=http")
    api_key: Optional[str] = kwargs.get("api_key")
    timeout: int = int(kwargs.get("timeout") or 120)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    fields = {"prompt": prompt}
    files = [("video", video_path)] if video_path.exists() else []
    # Optionally include a subset of frames
    for i, fpath in enumerate(frames[:5]):
        files.append((f"frame_{i:02d}", fpath))
    status, body = http_multipart_post(api_base, fields, files, headers, timeout)
    try:
        data = json.loads(body.decode("utf-8", errors="ignore"))
    except Exception:
        data = {"raw": body.decode("utf-8", errors="ignore")}
    return {"status": status, "response": data}


PROVIDERS = {
    "noop": analyze_noop,
    "http": analyze_http,
    # "openai": analyze_openai,  # to be implemented with provided docs/keys
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a video with an external vision API")
    parser.add_argument("video", type=str, help="Path to input video (mp4/webm)")
    parser.add_argument("--provider", type=str, default="noop", choices=sorted(PROVIDERS.keys()), help="Analysis provider")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: <video>_analysis.json)")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for extracted frames")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample for context")
    parser.add_argument("--prompt", type=str, default="Provide a structured analysis of the video content (scenes, motion, objects, actions).", help="Instruction for the vision model")
    parser.add_argument("--api-base", dest="api_base", type=str, default=None, help="Provider base URL (for provider=http)")
    parser.add_argument("--api-key-env", dest="api_key_env", type=str, default=None, help="Environment variable containing API key")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames on disk")
    parser.add_argument("--metadata-only", action="store_true", help="Skip API call; only write metadata + frame list")
    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    out_json = Path(args.output) if args.output else video_path.with_suffix("")
    if out_json.suffix != ".json":
        out_json = Path(str(out_json) + "_analysis.json")
    out_json = out_json.resolve()

    # Working directory
    if args.workdir:
        workdir = Path(args.workdir).expanduser().resolve()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        workdir = out_json.parent / f"analysis_{ts}_{uuid.uuid4().hex[:6]}"
    frames_dir = workdir / "frames"

    print("=" * 70)
    print("🎥 External Video Analyzer (scaffold)")
    print("=" * 70)
    print(f"📄 Video: {video_path}")

    # Gather metadata
    print("\n📊 Probing video metadata...")
    meta = ffprobe_metadata(video_path)
    if meta:
        print(f"   ✓ Detected: {meta}")
    else:
        print("   ⚠️ No metadata available (ffprobe missing?)")

    # Extract frames for context
    print("\n🖼️  Extracting frames for context...")
    duration = meta.get("duration") if isinstance(meta, dict) else None
    frames = extract_frames(video_path, frames_dir, max(1, args.frames), duration)
    print(f"   ✓ Extracted {len(frames)} frame(s) → {frames_dir}")

    # Resolve API key
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"   ⚠️ Env var {args.api_key_env} not set; proceeding without API key")

    result: Dict[str, Any]
    if args.metadata_only:
        print("\nℹ️  Skipping API call (metadata-only)")
        result = {"note": "metadata-only"}
    else:
        provider_fn = PROVIDERS.get(args.provider)
        if not provider_fn:
            print(f"❌ Unknown provider: {args.provider}")
            sys.exit(1)
        print(f"\n🔗 Calling provider: {args.provider}")
        try:
            result = provider_fn(
                video_path=video_path,
                frames=frames,
                prompt=args.prompt,
                api_base=args.api_base,
                api_key=api_key,
                timeout=args.timeout,
            )
        except Exception as e:
            print(f"   ❌ Provider error: {e}")
            result = {"error": str(e)}

    # Compose output payload
    payload = {
        "input": {
            "video_path": str(video_path),
            "size_bytes": video_path.stat().st_size if video_path.exists() else None,
            "metadata": meta,
            "sampled_frames": [str(p) for p in frames],
        },
        "provider": {
            "name": args.provider,
            "api_base": args.api_base,
        },
        "prompt": args.prompt,
        "analysis": result,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved analysis → {out_json}")

    if not args.keep_frames:
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass
    else:
        print(f"📁 Kept frames under: {workdir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

