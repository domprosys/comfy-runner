# AGENTS.md - AI Assistant Context Documentation

This document provides comprehensive context for AI coding assistants working on this project.

## Project Overview

**ComfyUI API Automation Suite** - A Python toolkit for automating ComfyUI workflows with advanced batch parameter exploration capabilities.

**Primary Goal:** Enable systematic exploration of video generation parameter spaces to discover optimal settings for WAN (video diffusion) models.

**Key Innovation:** Combines intelligent sampling strategies (Sobol sequences) with automated workflow execution to efficiently explore high-dimensional parameter spaces.

**Repository:** https://github.com/domprosys/comfy-runner

---

## What We Built

### Core Components (Packaged CLI)

1. **Single Workflow Runner** (`src/comfy_api/runner.py` → CLI: `cr-run`)
   - Executes a single ComfyUI workflow via API
   - HTTP polling (no WebSockets to avoid memory issues with large files)
   - Streaming file downloads to handle 100MB+ video files

2. **Workflow Analyzer** (`src/comfy_api/analyzer.py` → CLI: `cr-analyze`)
   - Automatically extracts modifiable parameters from any workflow JSON
   - Generates human-readable parameter names and suggested ranges
   - Outputs ready-to-edit YAML config templates

3. **Generic Batch Runner** (`src/comfy_api/batch.py` → CLI: `cr-batch`)
   - Works with ANY ComfyUI workflow (via parameter paths)
   - Sampling strategies: Sobol, LHS, Grid, Random
   - Resume capability with `progress.json`
   - Streaming downloads, organized output, metadata tracking

4. **Shot-Based Batch Runner** (`src/comfy_api/shots_batch.py` → CLI: `cr-shots`)
   - Tests multiple "shots" (scene variations) with different prompts and frame pairs
   - Injects shot-specific data into workflow and organizes outputs per shot

5. **Contact Sheet Generator** (`src/comfy_api/contact_sheet.py` → CLI: `cr-contact-sheet`)
   - Creates interactive HTML visualization of batch results (ffmpeg required)

---

## Key Design Decisions & Rationale

### 1. HTTP Polling vs WebSockets

**Decision:** Use HTTP polling instead of WebSockets for execution monitoring.

**Why:**
- WebSockets can freeze when downloading large files (100MB+ videos)
- Simpler error handling and connection management
- More reliable for long-running batches
- Trade-off: Less detailed progress updates, but much more stable

**Implementation:** Poll `/history/{prompt_id}` every 2 seconds until `outputs` appears.

### 2. Parameter Path System

**Decision:** Use dot-notation paths (`nodes.91.inputs.strength_model`) to modify workflows.

**Why:**
- Fully generic - works with any workflow structure
- No hardcoded node IDs
- Easy to validate (path exists or doesn't)
- Allows workflow updates without regenerating configs (if node IDs stay same)

**Example:**
```yaml
lora_strength:
  path: nodes.91.inputs.strength_model
  type: continuous
  min: 0.5
  max: 3.0
```

### 3. Sobol Sequences for Parameter Sampling

**Decision:** Use quasi-random Sobol sequences as default sampling strategy.

**Why:**
- 10x more efficient than random sampling for same coverage
- Excellent for high-dimensional spaces (4+ parameters)
- Progressive refinement (can stop early if needed)
- Critical for video generation where each run takes 5-15 minutes

**Math:** Sobol fills parameter space uniformly by maximizing distance between consecutive samples.

### 4. Two-Layer Seed Handling

**Decision:** Separate "parameter exploration" from "seed variation".

**Why:**
- Seeds are NOT part of searchable space (seed=1000 and seed=1001 are unrelated)
- Sobol explores continuous parameters (strength, cfg, etc.)
- `seeds_per_sample` generates random variations at each parameter point
- Avoids wasting sampling efficiency on meaningless seed exploration

**Implementation:**
```yaml
parameters:
  lora_strength: {min: 0.5, max: 3.0}  # Explored by Sobol
  seed: {type: random_seed}            # Generated randomly per run

seeds_per_sample: 3  # 3 random seeds per parameter combo
```

### 5. Streaming File Downloads

**Decision:** Download files in 8KB chunks, write directly to disk.

**Why:**
- Video files can be 500MB+
- Loading entire file into memory freezes systems
- Chunk-based streaming prevents OOM errors
- Shows progress for large files (>1MB)

**Implementation:** `urllib.request` with chunked reading, no intermediate buffering.

### 6. Shot-Based Organization

**Decision:** Separate runner for shot-based testing vs parameter exploration.

**Why:**
- Different use cases: testing scenes vs exploring parameters
- Shot runner injects prompt + images per shot
- Cleaner separation of concerns
- Could merge later, but keeping separate for clarity

---

## Architecture & Data Flow

### Single Workflow Execution

```
workflow.json → comfy_api.runner (cr-run)
                    ↓
            POST /prompt (ComfyUI API)
                    ↓
            Poll /history/{prompt_id}
                    ↓
            GET /view (download outputs)
                    ↓
            output/{prompt_id}/video.mp4
```

### Batch Parameter Exploration

```
workflow.json → comfy_api.analyzer (cr-analyze) → config.yaml
                                          ↓
                                      (user edits)
                                          ↓
config.yaml → comfy_api.batch (cr-batch)
                    ↓
            Generate parameter samples (Sobol/LHS/Grid)
                    ↓
            For each sample:
                Modify workflow JSON (inject parameters)
                Submit to ComfyUI
                Poll until complete
                Download outputs
                Save metadata
                    ↓
            output/batch_YYYY-MM-DD/
                runs/{run_id}/video.mp4
                metadata.csv
                progress.json
```

### Shot-Based Testing

```
shots/shot1/prompt.txt + images
                    ↓
        comfy_api.shots_batch (cr-shots)
                    ↓
        Copy images to ComfyUI/input/
        Inject prompt + image paths into workflow
        Generate parameter variations
                    ↓
        For each shot:
            For each parameter combo:
                Execute workflow
                    ↓
        output/batch_shots_YYYY-MM-DD/
            shot1/runs/{run_id}/video.mp4
            shot2/runs/{run_id}/video.mp4
            ...
```

---

## File Structure & Purpose

```
comfy-runner/
├── src/comfy_api/
│   ├── runner.py           # Single workflow execution (CLI: cr-run)
│   ├── analyzer.py         # Extract parameters, generate config (CLI: cr-analyze)
│   ├── batch.py            # Generic parameter exploration (CLI: cr-batch)
│   ├── shots_batch.py      # Shot-based testing (CLI: cr-shots)
│   └── contact_sheet.py    # Visualization tool (CLI: cr-contact-sheet)
├── workflows/
│   ├── wan2_14B_flf2v/
│   │   ├── workflow.json
│   │   └── configs/
│   │       ├── baseline.yaml
│   │       ├── smoke.yaml
│   │       └── shots.yaml
│   └── wan2_5B_ti2v/
│       └── ...
├── shots/                  # Shot folders (see shots/README.md)
├── README.md               # User documentation
├── AGENTS.md               # This file (AI context)
├── pyproject.toml          # Package + CLI entry points
└── output/                 # Generated results (gitignored)
```

### Legacy Files

- Root-level scripts (`comfy_runner.py`, `analyze_workflow.py`, `batch_runner_v2.py`, `batch_runner_shots.py`, `generate_contact_sheet.py`) exist for backward compatibility but the packaged CLIs are the primary interface.

---

## Important Context

### ComfyUI Workflow Structure

ComfyUI workflows are JSON graphs where:
- **Nodes** are keyed by ID strings (`"6"`, `"91"`, etc.)
- **Connections** are arrays: `["4", 0]` means "node 4, output 0"
- **Parameters** are in `node.inputs` dictionary

**Example:**
```json
{
  "91": {
    "inputs": {
      "strength_model": 2.0,
      "lora_name": "model.safetensors",
      "model": ["37", 0]  // Connection to node 37
    },
    "class_type": "LoraLoaderModelOnly"
  }
}
```

### Parameter Types Explained

1. **Continuous** - Sampled from range by Sobol/LHS
   ```yaml
   lora_strength:
     type: continuous
     min: 0.5
     max: 3.0
   ```

2. **Linear** - Discrete steps with fixed increment
   ```yaml
   cfg:
     type: linear
     min: 0.5
     max: 2.5
     step: 0.5  # [0.5, 1.0, 1.5, 2.0, 2.5]
   ```

3. **Values** - Specific discrete values only
   ```yaml
   shift:
     type: values
     values: [3, 4, 5, 6, 7]
   ```

4. **Random Seed** - Auto-generated each run
   ```yaml
   seed:
     type: random_seed
   ```

### WAN Workflow Specifics (workflows/wan2_14B_flf2v/workflow.json)

**Critical Parameters:**
- **Prompt:** Node 6 (`nodes.6.inputs.text`)
- **First Frame:** Node 68 (`nodes.68.inputs.image`)
- **Last Frame:** Node 62 (`nodes.62.inputs.image`)
- **Frame Count:** Node 67 (`nodes.67.inputs.length`)
- **LoRA Strengths:** Nodes 91, 92 (`strength_model`)
- **Sampling Shift:** Nodes 54, 55 (`shift`)
- **CFG:** Nodes 57, 58 (`cfg`)

**Two-Pass Architecture:**
- High-noise pass (nodes 57, 91) → creates initial structure
- Low-noise pass (nodes 58, 92) → refines details

---

## Known Issues & Future Work

### Known Issues

1. **Bug Fixed (2025-11-04):** `format_run_name()` had duplicate `run_id` argument
   - Fixed in batch runner (`src/comfy_api/batch.py`)
   - Old: `pattern.format(**format_params, run_id=run_id)`
   - New: `pattern.format(**format_params)`

2. **Disk Space:** Videos are stored twice (ComfyUI/output + batch output)
   - Planned: Add `delete_from_comfyui` option
   - Planned: Make ComfyUI path configurable (default: `~/comfy/ComfyUI`)

3. **Resume:** Not yet implemented for shot-based runner
   - Works fine for regular batch runner

### Future Enhancements

1. **File Management:**
   - Move files from ComfyUI output after download
   - Configurable ComfyUI install path
   - Symlinks instead of downloads for same-machine setups

2. **Shot System:**
   - Per-shot negative prompts
   - Resume support for shot batches
   - Cross-shot parameter comparison tools

3. **Analysis Tools:**
   - Automatic quality metrics (CLIP scores, etc.)
   - Best result selection
   - Parameter correlation analysis
   - Tensorboard integration

4. **Performance:**
   - Multi-GPU batch distribution
   - Queue management for multiple runners
   - Priority queue for important parameter combos

---

## Development Workflow

### Testing a Workflow

```bash
# 1. Test single execution first (critical!)
cr-run workflows/<name>/workflow.json

# 2. If works, analyze for parameters
cr-analyze workflows/<name>/workflow.json

# 3. Edit generated config (uncomment parameters to vary)
#    e.g., workflows/<name>/configs/baseline.yaml

# 4. Start with a small test batch
#    In the config: num_samples: 5, seeds_per_sample: 1

# 5. Run test
cr-batch workflows/<name>/configs/baseline.yaml

# 6. If successful, scale up (e.g., num_samples: 50, seeds_per_sample: 3)
```

### Typical Parameter Values

Based on testing with WAN models:

- **LoRA Strength:** 0.5 - 3.0 (sweet spot: 1.5 - 2.5)
- **CFG:** 0.5 - 2.5 (lower = more creative, higher = more adherent)
- **Steps:** 2-6 for LoRA-accelerated models, 15-30 for standard
- **Shift (SD3):** 3-7 (affects noise schedule)
- **Frame Count:** 60-180 frames typical (2.5-7.5 seconds @ 24fps)

### Code Style & Conventions

- **Minimal dependencies:** Prefer stdlib when possible
- **Streaming I/O:** Never load large files fully into memory
- **Fail gracefully:** Log errors, continue batch on failure
- **Resume-friendly:** Save progress frequently, atomic writes
- **Readable names:** `lora_strength` not `param_nodes_91_inputs_strength_model`
- **Emojis in output:** For visual clarity in terminal (user requested)

---

## User Context

**User Profile:**
- Video generation with ComfyUI
- Testing WAN (video diffusion) models
- Running on remote server via SSH/RustDesk
- Limited to local generation (no cloud API costs)
- Wants systematic parameter exploration to find optimal settings
- Multiple shots/scenes to test across same parameter space

**User's Environment:**
- Server: `~/comfy/ComfyUI` (ComfyUI install)
- Project: `~/comfy/comfy-runner` (this project)
- Python: 3.12 in venv
- No uv (user not familiar), using standard venv + pip
- Access: Remote desktop (RustDesk) for file transfer

**User's Workflow:**
1. Design workflow in ComfyUI UI
2. Export API format (File → Export API)
3. Analyze workflow → get config template
4. Edit config (select parameters to vary)
5. Run batch overnight (via screen session)
6. Review results via contact sheet
7. Iterate with refined parameter ranges

---

## Important Gotchas

### 1. Node ID Stability

If user adds/removes nodes in ComfyUI, node IDs can change!
- `nodes.91.inputs.strength_model` might become `nodes.95.inputs.strength_model`
- Config paths break
- **Solution:** Re-run analyzer to generate new config with updated IDs

### 2. Image File References

ComfyUI workflows reference images by filename only:
```json
"inputs": {"image": "start_frame.png"}
```

**These files must exist in `ComfyUI/input/` directory!**
- Not embedded in workflow
- Not paths, just filenames
- Batch runner doesn't verify existence upfront
- Missing images = batch fails all runs

### 3. Seed Handling

**Seeds are NOT parameters to explore with Sobol!**
- seed=1000 and seed=1001 are completely unrelated
- Use `type: random_seed` to auto-generate
- Use `seeds_per_sample` for variation at each parameter point
- Never put seeds in continuous parameter ranges

### 4. Memory & Disk

Video generation is resource-intensive:
- 100MB-500MB per video typical
- 5-15 minutes per video on good GPU
- 100 videos = 10-50GB, 8-25 hours
- Always start with small test batch (5-10 videos)

### 5. Resume Behavior

Resume works by checking `progress.json`:
- Completed runs skipped (even if files deleted!)
- If you want to re-run: delete `progress.json`
- Run names must be deterministic for resume to work
- Currently no partial-run resume (all or nothing per run)

---

## Testing Checklist for AI Assistants

When making changes, verify:

1. **Single workflow runs:** `cr-run workflows/<name>/workflow.json`
2. **Analyzer generates valid config:** `cr-analyze workflows/<name>/workflow.json`
3. **Small batch completes:** `num_samples: 3, seeds_per_sample: 1`
4. **Resume works:** Interrupt batch, restart, verify skip
5. **Parameter injection:** Check workflow JSON has modified values
6. **File downloads:** Videos actually saved, not empty
7. **Metadata accuracy:** CSV matches actual parameters used
8. **Shot system:** Images copied, prompts injected correctly

---

## Communication Style with User

User preferences:
- **No emojis** in code unless explicitly requested
- **Technical accuracy** over friendliness
- **Concise explanations** with examples
- **Terminal-friendly** output (emojis OK in CLI output for UX)
- **Show, don't tell** - code examples over descriptions
- **Question first** when ambiguous (use AskUserQuestion tool)

---

## Session Summary (What We Built)

Starting point: User had ComfyUI, wanted to automate workflow execution.

What we delivered:
1. ✅ Single workflow runner (HTTP, no WebSockets)
2. ✅ Workflow analyzer (auto-extract parameters)
3. ✅ Generic batch runner (works with any workflow)
4. ✅ Multiple sampling strategies (Sobol, LHS, Grid)
5. ✅ Shot-based testing (multi-scene variations)
6. ✅ Contact sheet visualization
7. ✅ Resume capability
8. ✅ Readable parameter names
9. ✅ Comprehensive documentation

Current status:
- User has batch running on remote server
- Testing WAN video generation parameters
- Exploring shot-based system for multiple scenes
- Ready for iterative refinement based on results

---

## Quick Reference Commands

```bash
# Test single workflow
cr-run workflows/<name>/workflow.json

# Analyze workflow
cr-analyze workflows/<name>/workflow.json

# Run parameter batch
cr-batch workflows/<name>/configs/<config>.yaml

# Run shot-based batch
cr-shots workflows/<name>/configs/shots.yaml

# Generate visualization
cr-contact-sheet output/batch_YYYY-MM-DD/

# Run in screen (for long batches)
screen -S batch
cr-batch workflows/<name>/configs/<config>.yaml
# Ctrl+A, D to detach

# Monitor progress
ls output/batch_*/runs/ | wc -l
du -sh output/batch_*
```

---

## Questions for New AI Assistants

Before starting work, consider:
1. Is the user's batch currently running? (Don't break it!)
2. Does this change affect `src/comfy_api/batch.py`? (Upload to server needed)
3. Will this work with their remote setup? (SSH, screen sessions, etc.)
4. Is this generic or workflow-specific? (Aim for generic)
5. How does this affect disk space? (Videos are huge)
6. Is there a simpler approach? (User prefers minimal dependencies)

---

**Last Updated:** 2025-11-05
**Project Status:** Active development, deployed on user's remote server
**Contact:** See parent README.md for user documentation

---

*This document is for AI coding assistants. For user documentation, see README.md*
