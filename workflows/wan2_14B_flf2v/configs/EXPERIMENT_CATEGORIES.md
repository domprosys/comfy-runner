# The 3-Category Experiment Framework

A systematic approach to optimizing ComfyUI video generation workflows.

---

## 🚀 Quick Start: Auto-Generate Templates

**NEW!** You can now automatically generate all category templates for any workflow:

```bash
# Analyze workflow and auto-generate Category 1/2/3 templates
cr-analyze workflows/your_workflow/workflow.json --generate-category-templates
```

This will:
1. **Detect workflow type** (base model, Light LoRA, or SNR-based)
2. **Auto-generate** all 3 category configs with appropriate:
   - Sampler lists (optimized for detected step count)
   - Parameter ranges
   - LoRA parameters (if detected)
3. **Save templates** to `workflows/your_workflow/configs/`

**For existing workflows**: Templates are already created - just use them!

**For new workflows**: Run the analyzer with `--generate-category-templates` flag.

---

## Overview

All batch experiments fall into **three grand categories**, each with a specific purpose and optimal strategy:

1. **🏄 Sampler/Scheduler Surfing** - Find the best numerical solver
2. **🔬 Parameter Search** - Optimize numeric parameters
3. **🎲 Seed Mining** - Generate variations and find hero seeds

**Key Principle**: Run experiments **in sequence**, using winners from each phase to inform the next.

---

## Category 1: 🏄 Sampler/Scheduler Surfing

### What Varies
- **Sampler** (euler, dpmpp_2m, res_3m_ode, etc.)
- **Scheduler** (normal, karras, exponential, bong_tangent)

### What's Fixed
- All numeric parameters (CFG, shift, steps, LoRA strengths)
- Seeds (use 2-3 for consistency check)
- Resolution, FPS, video length
- Prompts

### Purpose
- Find the best **numerical method** for solving the diffusion equation
- Compare convergence speed vs quality tradeoffs
- Identify sampler/scheduler compatibility
- Understand model-specific preferences

### Strategy
**Grid Search** (exhaustive testing)
- Test ALL sampler/scheduler combinations
- Use 2 seeds per combo (verify consistency)
- Fix everything else at sensible defaults

### Expected Output
With 5 samplers × 4 schedulers × 2 seeds = **40 videos** (~1-2 hours)

### When to Use
✅ **FIRST STEP** for any new:
- Workflow or model
- LoRA or checkpoint update
- Prompt or scene type

✅ When you suspect your current sampler is suboptimal

❌ Don't use if you already know the best sampler/scheduler

### Config File
```bash
cr-batch workflows/wan2_14B_flf2v/configs/category2_sampler_surfing_template.yaml
```

### Analysis
1. Generate contact sheet: `cr-contact-sheet output/batch_*/`
2. Sort by visual quality
3. Note generation time from `metadata.csv`
4. Identify winners (e.g., "res_3m_ode + karras")

### What You Learn
- Which sampler gives best quality
- Which scheduler works best with each sampler
- Speed vs quality tradeoffs
- Model consistency (tight variation = stable)

---

## Category 2: 🔬 Parameter Search

### What Varies
- **Numeric parameters**: CFG, shift, LoRA strengths, FPS, steps, etc.

### What's Fixed
- **Sampler** (winner from Category 1)
- **Scheduler** (winner from Category 1)
- Seeds (use 2-3)
- Prompts

### Purpose
- Find **optimal parameter values**
- Understand parameter sensitivities
- Identify parameter interactions
- Tune for quality/speed tradeoffs

### Strategy

**Grid Search** (2-3 parameters):
- Exhaustive testing of discrete values
- Good for understanding individual effects
- Example: CFG × shift = 5 × 5 = 25 combos

**Sobol Search** (4+ parameters):
- Efficient high-dimensional exploration
- 50-100 samples cover space well
- Better than random or grid for many parameters

### Expected Output
- **Grid**: 5 × 5 × 6 × 2 seeds = 300 videos (~3-8 hours)
- **Sobol**: 50 × 2 seeds = 100 videos (~2-4 hours)

### When to Use
✅ After Category 1 (you know best sampler/scheduler)

✅ When quality isn't satisfactory with defaults

✅ For production workflows (final tuning)

❌ Don't vary sampler/scheduler here - that's Category 1!

### Config Files
```bash
# Grid search (2-3 parameters)
cr-batch workflows/wan2_14B_flf2v/configs/category3_parameter_search_grid.yaml

# Sobol search (4+ parameters)
cr-batch workflows/wan2_14B_flf2v/configs/category3_parameter_search_sobol.yaml
```

### Analysis
1. Generate contact sheet
2. Look for **sweet spots** (e.g., "CFG 1.5-2.0 is best")
3. Identify **parameter interactions** (e.g., "high shift needs lower CFG")
4. Plot response surfaces if using Sobol

### What You Learn
- Optimal value ranges for each parameter
- Which parameters matter most
- Parameter interaction effects
- Sensitivity vs robustness

---

## Category 3: 🎲 Seed Mining

### What Varies
- **Random seed** ONLY

### What's Fixed
- **Everything else**: sampler, scheduler, all parameters
- Use ALL winners from Categories 1 & 2

### Purpose
- Find **"hero seeds"** that produce exceptional outputs
- Generate diverse variations of optimal settings
- Understand model variation/consistency
- Build a seed library for production

### Strategy
**Generate many variations** (20-200 seeds)
- All with identical settings except seed
- Sort by quality to find gems
- Document best seeds for reuse

### Expected Output
20 seeds × 1 combo = **20 videos** (~30-60 min)

### When to Use
✅ **FINAL STEP** after optimizing sampler/scheduler/parameters

✅ For final production renders

✅ When you need variations of a perfect setup

❌ Don't mine seeds with suboptimal settings!

### Config File
```bash
cr-batch workflows/wan2_14B_flf2v/configs/category1_seed_mining_template.yaml
```

### Analysis
1. Generate contact sheet
2. Manually review or use video analysis for quality ratings
3. **Record seed numbers** from best outputs' `params.json`
4. Create "hero seed library" for this prompt/scene

### What You Learn
- Which seeds produce best results
- Model consistency (tight cluster vs wide variation)
- Variation patterns (some seeds favor certain aesthetics)

---

## The Optimal Workflow: 3-Phase Sequence

### Phase 1: 🏄 Sampler/Scheduler Surfing
**Goal**: Find the best solver

```bash
cr-batch workflows/wan2_14B_flf2v/configs/category2_sampler_surfing_template.yaml
cr-contact-sheet output/batch_*/
```

**Result**: "res_3m_ode + karras is the winner!"

**Time**: ~1-2 hours

---

### Phase 2: 🔬 Parameter Search
**Goal**: Optimize parameters for the winning combo

**Edit config first** - Add your winners:
```yaml
sampler_name: {values: ["res_3m_ode"]}  # ← Winner from Phase 1
scheduler: {values: ["karras"]}         # ← Winner from Phase 1
```

```bash
cr-batch workflows/wan2_14B_flf2v/configs/category3_parameter_search_sobol.yaml
cr-contact-sheet output/batch_*/
```

**Result**: "Optimal CFG=1.75, shift=5.5, LoRA=2.0"

**Time**: ~2-4 hours (Sobol) or ~6-8 hours (Grid)

---

### Phase 3: 🎲 Seed Mining
**Goal**: Find hero seeds with optimal settings

**Edit config first** - Add ALL winners:
```yaml
sampler_name: {values: ["res_3m_ode"]}     # ← Winner
scheduler: {values: ["karras"]}            # ← Winner
sampler_cfg: {values: [1.75]}              # ← Optimal
sampling_shift: {values: [5.5]}            # ← Optimal
lora_strength_model: {values: [2.0]}       # ← Optimal
```

```bash
cr-batch workflows/wan2_14B_flf2v/configs/category1_seed_mining_template.yaml
cr-contact-sheet output/batch_*/
```

**Result**: "Seed 847392 is perfect for this prompt!"

**Time**: ~30-60 minutes

---

## Quick Reference Table

| Category | What Varies | What's Fixed | Strategy | Videos | Time | When |
|----------|-------------|--------------|----------|--------|------|------|
| **🏄 Surfing** | Sampler, Scheduler | Params, Seeds | Grid | 40 | 1-2h | **FIRST** |
| **🔬 Search** | Params (CFG, shift, etc) | Sampler, Scheduler | Grid/Sobol | 100-300 | 2-8h | **SECOND** |
| **🎲 Mining** | Seed ONLY | Everything | N/A | 20-200 | 0.5-3h | **FINAL** |

---

## Common Mistakes to Avoid

### ❌ Mining Seeds with Suboptimal Settings
Don't run Category 3 until you've completed Categories 1 & 2!

**Wrong**:
```yaml
# Mining with default euler + normal
sampler_name: {values: ["euler"]}     # ← Never tested if this is best!
seeds_per_sample: 100
```

**Right**:
```yaml
# Mining with proven winners
sampler_name: {values: ["res_3m_ode"]}     # ← Tested and won in Phase 1
scheduler: {values: ["karras"]}            # ← Tested and won in Phase 1
sampler_cfg: {values: [1.75]}              # ← Optimized in Phase 2
seeds_per_sample: 100
```

---

### ❌ Varying Sampler in Parameter Search
Don't mix categories!

**Wrong**:
```yaml
# Category 2 config (parameter search)
sampler_name: {values: ["euler", "dpmpp_2m"]}  # ← This is Category 1!
sampler_cfg: {min: 1.0, max: 3.0}              # ← This is Category 2!
```

**Right**:
```yaml
# Category 2: ONLY vary parameters
sampler_name: {values: ["res_3m_ode"]}  # ← Fixed (winner from Category 1)
sampler_cfg: {min: 1.0, max: 3.0}       # ← Vary this
```

---

### ❌ Using Too Many Seeds in Category 1 & 2
Save seed mining for Category 3!

**Wrong**:
```yaml
# Category 1 config (sampler surfing)
seeds_per_sample: 50  # ← TOO MANY! Wastes time
```

**Right**:
```yaml
# Category 1: Just 2 seeds for consistency check
seeds_per_sample: 2  # ← Enough to verify it's not a fluke
```

---

## Advanced Tips

### Iterative Refinement
After Phase 3, you might discover new insights. Loop back!

**Example**:
1. Phase 1: "res_3m_ode + karras wins"
2. Phase 2: Optimize parameters
3. Phase 3: Mine 100 seeds, discover "seed 12345 looks AMAZING"
4. **Return to Phase 2**: Maybe this seed reveals new optimal parameters?
5. Test a few parameter variations with seed 12345

### Multi-Prompt Workflows
Run all 3 phases for **each unique prompt/scene**:

```
Prompt A: Surfing → Search → Mining → Best combo A
Prompt B: Surfing → Search → Mining → Best combo B
Prompt C: Surfing → Search → Mining → Best combo C
```

Different prompts may prefer different samplers!

### Speed Optimization
If generation is slow, run Category 1 with **lower resolution** first:

```yaml
# Category 1 at 640x360 (faster)
wanfirstlastframetovideo_width: {values: [640]}
wanfirstlastframetovideo_height: {values: [360]}
```

Then verify winners at full resolution in Category 2.

---

## Analysis Workflow

### After Every Batch

1. **Generate contact sheet**:
   ```bash
   cr-contact-sheet output/batch_YYYY-MM-DD_HH-MM-SS/
   ```

2. **Open in browser**:
   ```bash
   firefox output/batch_*/contact_sheet.html
   ```

3. **Optional - Video quality analysis**:
   ```bash
   for video in output/batch_*/runs/*/video.mp4; do
     cr-video-analyze "$video" \
       --provider ali_openai_video \
       --model qwen3-omni-flash \
       --no-frames  # Faster
   done
   ```

4. **Examine metadata.csv**:
   ```python
   import pandas as pd
   df = pd.read_csv('output/batch_*/metadata.csv')

   # Average time by sampler
   print(df.groupby('sampler_name')['generation_time'].mean())

   # Best parameters
   # (correlate with your visual quality assessment)
   ```

---

## File Reference

| Category | Config File | Purpose |
|----------|-------------|---------|
| 🏄 Surfing | `category2_sampler_surfing_template.yaml` | Find best sampler/scheduler |
| 🔬 Search (Grid) | `category3_parameter_search_grid.yaml` | Optimize 2-3 parameters |
| 🔬 Search (Sobol) | `category3_parameter_search_sobol.yaml` | Optimize 4+ parameters |
| 🎲 Mining | `category1_seed_mining_template.yaml` | Generate variations |
| Quick Test | `example_category2_quick_test.yaml` | Fast 12-video test |

---

## Expected Time Investment

**Full 3-phase optimization**: ~5-12 hours total

| Phase | Videos | Time | Can Interrupt? |
|-------|--------|------|----------------|
| Surfing | 40 | 1-2h | ✅ Yes (resume) |
| Search | 100-300 | 2-8h | ✅ Yes (resume) |
| Mining | 20-200 | 0.5-3h | ✅ Yes (resume) |

**Quick validation**: ~30 minutes
- Use `example_category2_quick_test.yaml` (12 videos)

---

**Last Updated**: 2025-11-11
**For**: comfy-runner batch exploration system
