# ComfyUI API Automation Suite

A comprehensive Python toolkit for automating ComfyUI workflows via API, with advanced batch parameter exploration for discovering optimal generation settings.

## 🚀 Features

- **Single Workflow Execution** - Run any ComfyUI workflow from command line
- **Generic Workflow Analysis** - Automatically extract all parameters from any workflow
- **Intelligent Batch Exploration** - Test parameter variations with smart sampling strategies
- **Resume Capability** - Continue interrupted batch runs without re-running completed jobs
- **Visual Comparison** - Generate interactive HTML contact sheets to compare results
- **Zero WebSocket Dependencies** - Simple HTTP polling (no freezing issues with large files)
- **Streaming Downloads** - Memory-efficient handling of large video files

## 📦 Installation

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** PyYAML, SciPy, NumPy (for batch processing)

## 🎯 Quick Start

### 1. Run a Single Workflow

```bash
# Using package entry point
comfy-run workflows/wan2_14B_flf2v/workflow.json

# Or via module
python -m comfy_api.runner workflows/wan2_14B_flf2v/workflow.json
```

Output saved to `output/{prompt_id}/`.

### 2. Analyze a Workflow

Automatically extract all parameters and generate batch config template:

```bash
comfy-analyze workflows/wan2_14B_flf2v/workflow.json
```

This creates `my_workflow_batch_config.yaml` with:
- All numeric parameters detected
- Suggested exploration ranges
- Smart categorization (seeds, strengths, steps, etc.)

### 3. Configure Batch Exploration

Edit the generated config file:

```yaml
workflow_file: my_workflow.json
sampling_strategy: sobol  # sobol, lhs, grid, or random
num_samples: 100
seeds_per_sample: 3

parameters:
  param_nodes.57.inputs.cfg:
    path: nodes.57.inputs.cfg
    type: linear           # or continuous, values, random_seed
    min: 0.5
    max: 2.5
    step: 0.5
```

**Parameter Types:**

- `continuous`: `{min: X, max: Y}` - Sample from continuous range (for Sobol/LHS)
- `linear`: `{min: X, max: Y, step: Z}` - Discrete steps (1.0, 1.5, 2.0, ...)
- `values`: `[A, B, C]` - Specific values only
- `random_seed`: Automatically generates random seeds

### 4. Run Batch Exploration

```bash
comfy-batch workflows/wan2_14B_flf2v/configs/baseline.yaml
```

Progress is automatically saved - safe to interrupt and resume!

### 5. Generate Visual Comparison

```bash
comfy-contact-sheet output/batch_2024-11-03_19-30-00
```

Creates interactive HTML with:
- Thumbnail grid of all outputs
- Parameter overlays
- Sortable/filterable
- Click to play full video

## 🧠 Sampling Strategies

### Sobol Sequences (Recommended) ⭐
- **Best for:** Exploring high-dimensional spaces efficiently
- **Coverage:** 100 samples covers space better than 1000 random samples
- **Use when:** You have 4+ parameters to vary

### Latin Hypercube Sampling (LHS)
- **Best for:** Understanding individual parameter sensitivity
- **Coverage:** Ensures even coverage across each parameter dimension
- **Use when:** You want to study parameter effects independently

### Grid Search
- **Best for:** Exhaustive testing of discrete values
- **Warning:** Grows exponentially (5×4×3 = 60 combinations)
- **Use when:** You have 2-3 key parameters with few values each

### Random Sampling
- **Best for:** Baseline exploration
- **Coverage:** Least efficient but simplest
- **Use when:** Quick tests or comparing to other strategies

## 📊 Smart Exploration Workflow

**Phase 1: Coarse Exploration**
```bash
# Quick scan with 50 Sobol samples
num_samples: 50
seeds_per_sample: 1
```

**Phase 2: Review Results**
```bash
python generate_contact_sheet.py output/batch_xxx
# Sort by parameters, identify interesting regions
```

**Phase 3: Refined Exploration**
```bash
# Narrow ranges based on Phase 1 findings
parameters:
  lora_strength:
    min: 1.5  # Refined from promising 1.0-2.0 range
    max: 2.5
```

**Phase 4: Production Grid**
```bash
# Exhaustive grid on optimal ranges
sampling_strategy: grid
parameters:
  lora_strength: {values: [1.5, 2.0, 2.5]}
  cfg: {values: [1.0, 1.5]}
```

## 📁 Output Structure

```
output/batch_2024-11-03_19-30-00/
├── config.yaml              # Config snapshot
├── metadata.csv             # All parameters + results (import to Excel/Pandas)
├── progress.json            # Resume tracking
├── contact_sheet.html       # Interactive visualization
├── thumbnails/              # Auto-extracted frames
│   └── thumb_0001.jpg
└── runs/
    ├── 0001_lora1.5_cfg1.0/
    │   ├── video.mp4
    │   └── params.json      # Exact parameters used
    └── 0002_lora2.0_cfg1.5/
        └── ...
```

## 🔧 Advanced Configuration

### Custom Run Naming

```yaml
output:
  run_name_pattern: "{run_id:04d}_lora{lora_strength:.1f}_cfg{cfg:.1f}"
  # Creates: 0001_lora1.5_cfg1.0/
```

### Resume Interrupted Batches

```yaml
resume:
  enabled: true      # Default: true
  skip_existing: true  # Skip if output already exists
```

Just re-run the same command - completed runs are skipped automatically!

### Multiple Workflows

Analyze and run different workflows:

```bash
# Text-to-image
python analyze_workflow.py txt2img_workflow.json
python batch_runner_v2.py txt2img_workflow_batch_config.yaml

# Image-to-image
python analyze_workflow.py img2img_workflow.json
python batch_runner_v2.py img2img_workflow_batch_config.yaml
```

## 🎓 Example: LoRA Strength Exploration

```yaml
workflow_file: video_wan2_2_14B_flf2v.json
sampling_strategy: linear
num_samples: 10  # Ignored for linear strategy
seeds_per_sample: 5  # 5 variations per strength value

parameters:
  lora_high_strength:
    path: nodes.91.inputs.strength_model
    type: linear
    min: 0.5
    max: 3.0
    step: 0.5
    # Results in: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

  lora_low_strength:
    path: nodes.92.inputs.strength_model
    type: linear
    min: 0.5
    max: 3.0
    step: 0.5

  seed:
    path: nodes.57.inputs.noise_seed
    type: random_seed  # Auto-generated per run
```

This creates 6×6×5 = **180 videos** exploring all LoRA strength combinations with 5 seed variations each.

## 🎯 Parameter Discovery Tips

**The analyzer detects:**
- ✅ Sampling parameters (steps, cfg, schedulers)
- ✅ Model parameters (shift, denoise, strength)
- ✅ Seeds (automatically tagged as `random_seed`)
- ✅ Resolution (width, height, length)
- ✅ LoRA/model strengths
- ✅ Text prompts (flagged but not varied by default)
- ✅ File references (models, VAEs, images)

**Good parameters to explore:**
- `cfg` (guidance scale) - 0.5 to 3.0
- `steps` - Model-dependent, usually 10-50
- `lora_strength` - 0.5 to 3.0
- `shift` (for SD3/Flux models) - 3 to 7
- `denoise` - 0.8 to 1.0 for variations

**Parameters to fix:**
- Resolution (expensive to vary)
- Model files (unless comparing models)
- Text prompts (vary manually or use prompt templates)

## 🐛 Troubleshooting

**"Connection refused"**
- Make sure ComfyUI is running on `127.0.0.1:8188`
- Check `comfyui.server_address` in config

**"Workflow file not found"**
- Use absolute paths or place workflow in same directory
- Export workflow via ComfyUI: `File -> Export (API)`

**Batch runs out of disk space**
- Videos can be large! Monitor `output/` directory
- Consider fewer `seeds_per_sample`
- Delete failed runs: `rm -rf output/batch_xxx/runs/*/` (keep metadata.csv)

**Contact sheet missing videos**
- Requires `ffmpeg` installed for thumbnail extraction
- Install: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)

## 📚 File Reference

| File | Purpose | Use When |
|------|---------|----------|
| `src/comfy_api/runner.py` | Single workflow execution | Testing workflows, one-off generation |
| `src/comfy_api/analyzer.py` | Parameter extraction | Starting a new batch exploration project |
| `src/comfy_api/batch.py` | Generic batch executor | Running parameter exploration |
| `legacy/batch_config.yaml` | Old hardcoded config | (Deprecated - use analyzer instead) |
| `legacy/batch_runner.py` | Old hardcoded runner | (Deprecated - use v2 instead) |
| `generate_contact_sheet.py` | Visualization | Comparing batch results |

## 🚦 Workflow

```
1. Export workflow from ComfyUI (File -> Export API)
   ↓
2. comfy-analyze workflows/<name>/workflow.json
   ↓
3. Edit workflow_batch_config.yaml (uncomment parameters)
   ↓
4. comfy-batch workflows/<name>/configs/<config>.yaml
   ↓
5. comfy-contact-sheet output/batch_xxx

## 📁 Repository Layout

```
workflows/
  wan2_14B_flf2v/
    workflow.json
    configs/
      baseline.yaml
      shots.yaml
  wan2_5B_ti2v/
    workflow.json
    configs/
shots/
src/comfy_api/
docs/
legacy/
output/  (gitignored)
```
   ↓
6. Open contact_sheet.html in browser
```

## 📖 Background

### Why Sobol Sequences?

Traditional approaches:
- **Random sampling:** Inefficient, clusters and gaps in space
- **Grid search:** Exponential growth, infeasible for 4+ parameters

**Sobol sequences:**
- Quasi-random low-discrepancy sequences
- Fills space uniformly with minimal samples
- 100 Sobol samples ≈ 10,000 random samples in coverage

**Math:** Uses prime number bases to generate points that avoid clustering. Each new sample maximizes distance from previous samples.

### Why HTTP Polling Instead of WebSockets?

WebSockets are great for real-time progress, but:
- Can freeze when downloading large files (>100MB videos)
- Requires managing connection state
- More complex error handling

HTTP polling:
- Simple and robust
- No connection state to manage
- Works reliably with streaming downloads
- Easier to debug

Trade-off: Less detailed progress updates, but more reliable execution.

## 🤝 Contributing

Suggestions welcome! This is a personal project for exploring ComfyUI parameter spaces.

**Future ideas:**
- Prompt templating/variation
- Multi-GPU batch distribution
- Result quality metrics (CLIP scores, etc.)
- Automatic "best" result selection
- Tensorboard integration

## 📄 License

MIT - Use freely for personal or commercial projects.

---

**Happy exploring! 🚀**

*Questions? Issues? The code is the documentation - it's well-commented!*
