#!/usr/bin/env python3
"""
Generate Contact Sheet Visualization (packaged)
Creates an HTML grid view of batch-generated videos with parameter overlays.
"""

import json
import sys
import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Any


def extract_thumbnail(video_path: Path, output_path: Path, timestamp: str = "00:00:01") -> bool:
    try:
        cmd = [
            'ffmpeg', '-ss', timestamp, '-i', str(video_path), '-vframes', '1', '-q:v', '2', '-y', str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"   ⚠️  Failed to extract thumbnail from {video_path.name}: {e}")
        return False


def load_metadata(batch_dir: Path) -> List[Dict[str, Any]]:
    metadata_file = batch_dir / "metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get('status') == 'success']


def generate_thumbnails(batch_dir: Path, metadata: List[Dict[str, Any]]) -> Dict[str, Path]:
    thumbnails_dir = batch_dir / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)
    print(f"\n🖼️  Generating thumbnails...")
    thumbnails = {}
    for i, row in enumerate(metadata, 1):
        run_dir = Path(row['output_path'])
        run_id = row['run_id']
        video_files = list(run_dir.glob("*.mp4")) + list(run_dir.glob("*.webm"))
        if not video_files:
            print(f"   [{i}/{len(metadata)}] ⚠️  No video found in {run_dir.name}")
            continue
        video_path = video_files[0]
        thumb_path = thumbnails_dir / f"thumb_{run_id}.jpg"
        if thumb_path.exists():
            print(f"   [{i}/{len(metadata)}] ⏭️  Using cached: {thumb_path.name}")
        else:
            print(f"   [{i}/{len(metadata)}] 📸 Extracting: {video_path.name}")
            success = extract_thumbnail(video_path, thumb_path)
            if not success:
                continue
        thumbnails[run_id] = thumb_path
    print(f"   ✓ Generated {len(thumbnails)} thumbnails")
    return thumbnails


def generate_html(batch_dir: Path, metadata: List[Dict[str, Any]], thumbnails: Dict[str, Path]) -> Path:
    output_file = batch_dir / "contact_sheet.html"
    # Use a template with placeholders to safely inject JSON
    template = """<!DOCTYPE html>
<html>
<head>
    <meta charset=\"UTF-8\">
    <title>ComfyUI Batch Contact Sheet</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 32px; margin-bottom: 10px; }
        .header .stats { color: #888; font-size: 14px; }
        .controls { background: #2a2a2a; padding: 20px; border-radius: 8px; margin-bottom: 30px; display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }
        .controls label { color: #aaa; margin-right: 10px; }
        .controls select, .controls input { background: #1a1a1a; color: #fff; border: 1px solid #444; padding: 8px 12px; border-radius: 4px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .item { background: #2a2a2a; border-radius: 8px; overflow: hidden; transition: transform 0.2s; cursor: pointer; }
        .item:hover { transform: scale(1.02); box-shadow: 0 8px 16px rgba(0,0,0,0.4); }
        .item img { width: 100%; height: 200px; object-fit: cover; display: block; }
        .item .info { padding: 15px; }
        .item .run-id { font-size: 12px; color: #666; margin-bottom: 8px; }
        .item .params { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; }
        .item .param { display: flex; justify-content: space-between; }
        .item .param .label { color: #888; }
        .item .param .value { color: #fff; font-weight: 500; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; align-items: center; justify-content: center; }
        .modal.active { display: flex; }
        .modal video { max-width: 90%; max-height: 90%; border-radius: 8px; }
        .modal-close { position: absolute; top: 20px; right: 40px; font-size: 40px; color: #fff; cursor: pointer; }
    </style>
    </head>
<body>
    <div class=\"header\">
        <h1>🎬 ComfyUI Batch Results</h1>
        <div class=\"stats\">
            <span id=\"total-items\">0</span> results •
            <span id=\"batch-dir\"></span>
        </div>
    </div>
    <div class=\"controls\">
        <div>
            <label>Sort by:</label>
            <select id=\"sort-select\">
                <option value=\"run_id\">Run ID</option>
                <option value=\"lora_strength\">LoRA Strength</option>
            </select>
        </div>
        <div>
            <label>Filter:</label>
            <input type=\"text\" id=\"filter-input\" placeholder=\"Search parameters...\">
        </div>
    </div>
    <div class=\"grid\" id=\"grid\"></div>
    <div class=\"modal\" id=\"modal\">
        <span class=\"modal-close\" onclick=\"closeModal()\">&times;</span>
        <video id=\"modal-video\" controls autoplay loop></video>
    </div>
    <script>
        const data = ___DATA_JSON___;
        const batchDir = ___BATCH_DIR_JSON___;
        document.getElementById('batch-dir').textContent = batchDir;
        document.getElementById('total-items').textContent = data.length;
        function renderGrid(items) {
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            items.forEach(item => {
                const thumbPath = `thumbnails/thumb_${item.run_id}.jpg`;
                const videoPath = `${item.output_path}`;
                const div = document.createElement('div');
                div.className = 'item';
                div.onclick = () => openVideo(videoPath);
                div.innerHTML = `
                    <img src=\"${thumbPath}\" alt=\"Run ${item.run_id}\">\n                    <div class=\"info\">\n                        <div class=\"run-id\">Run #${item.run_id}</div>\n                        <div class=\"params\">\n                            <div class=\"param\">\n                                <span class=\"label\">LoRA:</span>\n                                <span class=\"value\">${item.lora_strength || 'N/A'}</span>\n                            </div>\n                            <div class=\"param\">\n                                <span class=\"label\">CFG:</span>\n                                <span class=\"value\">${item.cfg || 'N/A'}</span>\n                            </div>\n                            <div class=\"param\">\n                                <span class=\"label\">Shift:</span>\n                                <span class=\"value\">${item.shift || 'N/A'}</span>\n                            </div>\n                            <div class=\"param\">\n                                <span class=\"label\">Seed:</span>\n                                <span class=\"value\">${item.seed || 'N/A'}</span>\n                            </div>\n                        </div>\n                    </div>\n                `;
                grid.appendChild(div);
            });
        }
        function openVideo(path) {
            const modal = document.getElementById('modal');
            const video = document.getElementById('modal-video');
            const runDir = path.split('/').pop();
            video.src = `runs/${runDir}/${runDir.split('_')[0]}.mp4`;
            modal.classList.add('active');
        }
        function closeModal() {
            const modal = document.getElementById('modal');
            const video = document.getElementById('modal-video');
            modal.classList.remove('active');
            video.pause();
            video.src = '';
        }
        document.getElementById('sort-select').addEventListener('change', (e) => {
            const sortBy = e.target.value;
            const sorted = [...data].sort((a, b) => {
                const aVal = parseFloat(a[sortBy]) || a[sortBy];
                const bVal = parseFloat(b[sortBy]) || b[sortBy];
                return aVal > bVal ? 1 : -1;
            });
            renderGrid(sorted);
        });
        document.getElementById('filter-input').addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = data.filter(item => JSON.stringify(item).toLowerCase().includes(query));
            renderGrid(filtered);
        });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
        renderGrid(data);
    </script>
</body>
</html>
"""
    data_json = json.dumps(metadata, indent=8)
    batch_dir_json = json.dumps(str(batch_dir.name))
    html = template.replace('___DATA_JSON___', data_json).replace('___BATCH_DIR_JSON___', batch_dir_json)
    with open(output_file, 'w') as f:
        f.write(html)
    return output_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m comfy_api.contact_sheet <batch_directory>")
        sys.exit(1)
    batch_path = Path(sys.argv[1])
    if not batch_path.exists():
        print(f"❌ Error: Batch directory not found: {batch_path}")
        sys.exit(1)
    print("=" * 70)
    print("🖼️  Contact Sheet Generator")
    print("=" * 70)
    print(f"\n📁 Batch directory: {batch_path}")
    print(f"\n📊 Loading metadata...")
    metadata = load_metadata(batch_path)
    print(f"   ✓ Loaded {len(metadata)} successful runs")
    thumbnails = generate_thumbnails(batch_path, metadata)
    print(f"\n🌐 Generating HTML contact sheet...")
    html_file = generate_html(batch_path, metadata, thumbnails)
    print(f"   ✓ Generated: {html_file}")
    print("\n" + "=" * 70)
    print("✅ Complete!")
    print("=" * 70)
    print(f"\n💡 Open in browser:")
    print(f"   file://{html_file.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
