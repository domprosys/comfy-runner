#!/usr/bin/env python3
"""
ComfyUI Workflow Runner (packaged)
Connects to a local ComfyUI instance and executes workflows via API.
Uses HTTP polling (no WebSockets) to monitor execution.
"""

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path


SERVER_ADDRESS = "127.0.0.1:8188"
OUTPUT_DIR = Path("output")
POLL_INTERVAL = 2


def queue_prompt(prompt):
    payload = {"prompt": prompt}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    response = urllib.request.urlopen(req)
    result = json.loads(response.read())
    return result['prompt_id']


def get_history(prompt_id):
    url = f"http://{SERVER_ADDRESS}/history/{prompt_id}"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError:
        return {}


def is_execution_complete(prompt_id):
    history = get_history(prompt_id)
    if prompt_id in history:
        prompt_data = history[prompt_id]
        return 'outputs' in prompt_data or 'status' in prompt_data
    return False


def poll_until_complete(prompt_id):
    print(f"\n🚀 Execution started (prompt_id: {prompt_id})")
    print("=" * 60)
    print("⏳ Polling for completion", end="", flush=True)

    dots = 0
    while not is_execution_complete(prompt_id):
        time.sleep(POLL_INTERVAL)
        dots = (dots + 1) % 4
        print(f"\r⏳ Polling for completion{'.' * dots}{' ' * (3 - dots)}", end="", flush=True)

    print(f"\r✓ Execution complete!{' ' * 30}")
    print("=" * 60)


def download_file_streaming(filename, subfolder, folder_type, save_path):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_params = urllib.parse.urlencode(params)
    url = f"http://{SERVER_ADDRESS}/view?{url_params}"

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(save_path, 'wb') as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 1_000_000:
                    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                    mb_downloaded = downloaded / 1_000_000
                    mb_total = total_size / 1_000_000
                    print(f"\r    Progress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.1f}%)", end="", flush=True)
        if total_size > 1_000_000:
            print()


def save_outputs(prompt_id):
    print(f"\n📥 Retrieving outputs...")
    history = get_history(prompt_id)
    if prompt_id not in history:
        print(f"⚠️  No history found for prompt_id: {prompt_id}")
        return []

    outputs = history[prompt_id].get('outputs', {})
    saved_files = []
    output_path = OUTPUT_DIR / prompt_id
    output_path.mkdir(parents=True, exist_ok=True)

    for _, node_output in outputs.items():
        if 'images' in node_output:
            for img in node_output['images']:
                filename = img['filename']
                subfolder = img.get('subfolder', '')
                file_type = img.get('type', 'output')
                print(f"  Downloading: {filename}")
                save_path = output_path / filename
                download_file_streaming(filename, subfolder, file_type, save_path)
                saved_files.append(save_path)
                print(f"  ✓ Saved: {save_path}")
        for media_key in ['gifs', 'videos']:
            if media_key in node_output:
                for media in node_output[media_key]:
                    filename = media['filename']
                    subfolder = media.get('subfolder', '')
                    file_type = media.get('type', 'output')
                    print(f"  Downloading: {filename}")
                    save_path = output_path / filename
                    download_file_streaming(filename, subfolder, file_type, save_path)
                    saved_files.append(save_path)
                    print(f"  ✓ Saved: {save_path}")

    return saved_files


def run_workflow(workflow_path):
    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        print(f"❌ Error: Workflow file not found: {workflow_path}")
        sys.exit(1)

    print(f"📄 Loading workflow: {workflow_file}")
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)

    print(f"🔗 Connecting to ComfyUI at {SERVER_ADDRESS}")
    try:
        print(f"📤 Submitting workflow...")
        prompt_id = queue_prompt(workflow)
        poll_until_complete(prompt_id)
        saved_files = save_outputs(prompt_id)
        print(f"\n✅ Complete! Generated {len(saved_files)} file(s)")
        for file_path in saved_files:
            print(f"   • {file_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m comfy_api.runner <workflow.json>")
        sys.exit(1)
    run_workflow(sys.argv[1])


if __name__ == "__main__":
    main()

