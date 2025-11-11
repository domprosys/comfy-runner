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
import textwrap
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


def _jsonable(obj):
    """Best-effort conversion of SDK objects to plain JSON-serializable data."""
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    # Dicts
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    # OpenAI pydantic-style objects
    for method_name in ("model_dump", "to_dict", "dict"):
        m = getattr(obj, method_name, None)
        if callable(m):
            try:
                return _jsonable(m())
            except Exception:
                pass
    # Fallback to __dict__ or str
    if hasattr(obj, "__dict__"):
        try:
            return _jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def _extract_text_from_analysis(analysis: Any) -> str:
    """Try to extract a human-readable text summary from various provider results."""
    # Direct string or 'text' field
    if isinstance(analysis, str):
        return analysis
    if isinstance(analysis, dict):
        if analysis.get("text"):
            return str(analysis.get("text"))
        # OpenAI-like structure with choices
        choices = analysis.get("choices")
        if isinstance(choices, list) and choices:
            parts = []
            for ch in choices:
                # ch may be dict with message.content
                msg = ch.get("message") if isinstance(ch, dict) else None
                content = None
                if isinstance(msg, dict):
                    content = msg.get("content")
                if content:
                    parts.append(str(content))
            if parts:
                return "\n\n".join(parts)
        # Sometimes nested under 'response'
        resp = analysis.get("response")
        if isinstance(resp, dict):
            sub = _extract_text_from_analysis(resp)
            if sub:
                return sub
    return ""


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


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def analyze_ali_openai(video_path: Path, frames: List[Path], prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Alibaba Cloud Model Studio via OpenAI-compatible SDK (DashScope compatible mode).

    Args (kwargs):
      - model: str (required)
      - base_url: str (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
      - api_key: str (required via --api-key-env ideally)
      - timeout: int (unused; SDK handles)
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai SDK not installed. Run: pip install -U openai") from e

    model = kwargs.get("model")
    if not model:
        raise ValueError("model is required for provider=ali_openai (e.g., qwen-vl-max)")
    # Default to Singapore (intl) base URL
    base_url = kwargs.get("base_url") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key = kwargs.get("api_key")
    if not api_key:
        raise ValueError("api_key is required (use --api-key-env to provide it)")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Build multimodal content: prompt + a handful of frames as data URLs
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    # Limit images to keep payload manageable
    for fpath in frames[:5]:
        try:
            with open(fpath, "rb") as f:
                data_url = f"data:image/jpeg;base64,{_b64(f.read())}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        except Exception:
            continue

    messages = [{"role": "user", "content": content}]

    # Use Chat Completions for compatibility
    resp = client.chat.completions.create(model=model, messages=messages)
    out = {
        "id": getattr(resp, "id", None),
        "choices": [],
        "usage": getattr(resp, "usage", None),
        "model": getattr(resp, "model", model),
    }
    try:
        for ch in resp.choices:
            msg = getattr(ch, "message", None)
            out["choices"].append({
                "index": getattr(ch, "index", None),
                "finish_reason": getattr(ch, "finish_reason", None),
                "content": getattr(msg, "content", None) if msg else None,
                "role": getattr(msg, "role", None) if msg else None,
            })
    except Exception:
        pass
    return out


def analyze_ali_openai_video(video_path: Path, frames: List[Path], prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Alibaba Cloud Model Studio via OpenAI-compatible SDK, sending a local video file
    as a data URL (base64) in a streaming Chat Completions request.

    Args (kwargs):
      - model: str (required), e.g., qwen3-omni-flash
      - base_url: str (default: Singapore https://dashscope-intl.aliyuncs.com/compatible-mode/v1)
      - api_key: str (required)
      - modalities: List[str] or comma-separated str (default: ["text"])
      - voice: str (when audio requested)
      - audio_format: str (e.g., wav) (when audio requested)
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai SDK not installed. Run: pip install -U openai") from e

    model = kwargs.get("model")
    if not model:
        raise ValueError("model is required for provider=ali_openai_video (e.g., qwen3-omni-flash)")
    base_url = kwargs.get("base_url") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    api_key = kwargs.get("api_key")
    if not api_key:
        raise ValueError("api_key is required (use --api-key-env to provide it)")

    # Build video data URL (note: reads the file into memory)
    mime = "video/mp4" if video_path.suffix.lower() in {".mp4", ".m4v"} else "video/webm"
    with open(video_path, "rb") as f:
        data_url = f"data:{mime};base64,{_b64(f.read())}"

    modalities = kwargs.get("modalities") or ["text"]
    if isinstance(modalities, str):
        modalities = [m.strip() for m in modalities.split(",") if m.strip()]
    voice = kwargs.get("voice") or "Cherry"
    audio_format = kwargs.get("audio_format") or "wav"

    client = OpenAI(base_url=base_url, api_key=api_key)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if modalities:
        payload["modalities"] = modalities
        if "audio" in modalities:
            payload["audio"] = {"voice": voice, "format": audio_format}

    # Stream and accumulate text
    text_parts: List[str] = []
    usage_obj = None
    stream = client.chat.completions.create(**payload)
    for chunk in stream:
        try:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # Prefer explicit content field over object repr
                content = None
                if hasattr(delta, "content"):
                    content = getattr(delta, "content")
                elif isinstance(delta, dict):
                    content = delta.get("content")
                # Append only real text
                if content:
                    text_parts.append(str(content))
            else:
                usage_obj = getattr(chunk, "usage", None)
        except Exception:
            # Ignore malformed chunks
            pass

    return {"text": "".join(text_parts), "usage": usage_obj, "model": model}


PROVIDERS = {
    "noop": analyze_noop,
    "http": analyze_http,
    "ali_openai": analyze_ali_openai,
    "ali_openai_video": analyze_ali_openai_video,
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a video with an external vision API")
    parser.add_argument("video", type=str, help="Path to input video (mp4/webm)")
    parser.add_argument("--provider", type=str, default="noop", choices=sorted(PROVIDERS.keys()), help="Analysis provider")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: <video>_analysis.json)")
    parser.add_argument("--workdir", type=str, default=None, help="Working directory for extracted frames")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to sample for context")
    parser.add_argument("--prompt", type=str, default="""You are analyzing a video generated by an AI video generation model. Provide a detailed technical analysis as if describing it to a film director who cannot see the footage.

PART 1 - CONTENT DESCRIPTION:
Describe what happens in the video: the setting, subjects, actions, camera movement, and overall narrative. Include cinematic details about lighting, composition, and atmosphere.

PART 2 - QUALITY ANALYSIS WITH RATINGS:
Evaluate the technical quality and realism on a 0-10 scale for each criterion:

1. MOTION & PHYSICS REALISM (0-10):
   - Do movements obey natural physics?
   - Any sliding, floating, or impossible actions?
   - Do objects and people move realistically?
   - Rate: [score]/10
   - Issues: [specific problems with timestamps]

2. CHARACTER CONSISTENCY (0-10):
   - Do people maintain consistent facial features throughout?
   - Any morphing of face, body proportions, or appearance?
   - Does identity remain stable?
   - Rate: [score]/10 (or N/A if no characters)
   - Issues: [specific problems with timestamps]

3. TEMPORAL COHERENCE (0-10):
   - Is motion smooth frame-to-frame?
   - Any flickering, jumping, or abrupt changes?
   - Does time flow naturally?
   - Rate: [score]/10
   - Issues: [specific problems with timestamps]

4. VISUAL FIDELITY (0-10):
   - Any blurriness, warping, distortion?
   - Are textures and details realistic?
   - Any "synthetic" or "AI-generated" visual qualities?
   - Rate: [score]/10
   - Issues: [specific problems with timestamps]

5. OVERALL REALISM (0-10):
   - Could this pass as conventionally filmed footage?
   - Would the average viewer detect it as AI-generated?
   - Rate: [score]/10
   - Summary: [overall assessment]

Use this rating scale:
10 = Perfect, indistinguishable from real footage
8-9 = Minor imperfections only experts would notice
6-7 = Noticeable issues but still convincing
4-5 = Obvious problems that break immersion
2-3 = Severe quality issues
0-1 = Completely unrealistic, unusable""", help="Instruction for the vision model")
    parser.add_argument("--api-base", dest="api_base", type=str, default=None, help="Provider base URL (for provider=http, or OpenAI-compatible base for ali_openai)")
    parser.add_argument("--api-key-env", dest="api_key_env", type=str, default="DASHSCOPE_API_KEY", help="Environment variable containing API key (default: DASHSCOPE_API_KEY)")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--model", type=str, default=None, help="Model name (required for ali_openai, e.g., qwen-vl-max)")
    parser.add_argument("--modalities", type=str, default="text", help="For ali_openai_video: 'text' or 'text,audio'")
    parser.add_argument("--voice", type=str, default="Cherry", help="For ali_openai_video when audio is requested")
    parser.add_argument("--audio-format", dest="audio_format", type=str, default="wav", help="For ali_openai_video when audio is requested")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames on disk")
    parser.add_argument("--metadata-only", action="store_true", help="Skip API call; only write metadata + frame list")
    parser.add_argument("--dotenv", type=str, default=".env", help="Optional path to .env to load")
    parser.add_argument("--no-frames", action="store_true", help="Do not extract frames (faster, less context)")
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
    frames: List[Path] = []
    if args.no_frames:
        print("\n🖼️  Skipping frame extraction (--no-frames)")
    else:
        print("\n🖼️  Extracting frames for context...")
        duration = meta.get("duration") if isinstance(meta, dict) else None
        frames = extract_frames(video_path, frames_dir, max(1, args.frames), duration)
        print(f"   ✓ Extracted {len(frames)} frame(s) → {frames_dir}")

    # Optionally load .env
    if args.dotenv:
        load_dotenv_if_present(args.dotenv)

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
                model=args.model,
                modalities=args.modalities,
                voice=args.voice,
                audio_format=args.audio_format,
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
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved analysis → {out_json}")

    # Pretty-print a clean text summary if available
    summary = _extract_text_from_analysis(result)
    if summary:
        # Save text summary to .txt file with same basename as video
        out_txt = video_path.with_suffix(".txt")
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(summary.strip())
            print(f"✅ Saved text summary → {out_txt}")
        except Exception as e:
            print(f"⚠️  Could not save text summary: {e}")

        print("\n🧾 Analysis Summary")
        print("=" * 70)
        wrapped = textwrap.fill(summary.strip(), width=100, replace_whitespace=False)
        print(wrapped)
        print("=" * 70)
    else:
        print("\nℹ️  No plain text content found in provider response.")

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
