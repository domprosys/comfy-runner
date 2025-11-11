# Sampler/Scheduler Guide for Different WAN2.2 Workflows

## 🎯 Quick Start: The 3-Category Experiment Framework

**NEW!** We now have a systematic approach to batch exploration:

1. **🏄 Category 2: Sampler/Scheduler Surfing** - Find the best solver
2. **🔬 Category 3: Parameter Search** - Optimize numeric parameters
3. **🎲 Category 1: Seed Mining** - Generate variations and find hero seeds

**See**: [`EXPERIMENT_CATEGORIES.md`](./EXPERIMENT_CATEGORIES.md) for the complete framework and workflow.

**Ready-to-use configs**:
- Quick test (12 videos): [`example_category2_quick_test.yaml`](./example_category2_quick_test.yaml)
- Sampler surfing (40 videos): [`category2_sampler_surfing_template.yaml`](./category2_sampler_surfing_template.yaml)
- Parameter search: [`category3_parameter_search_grid.yaml`](./category3_parameter_search_grid.yaml) or [`category3_parameter_search_sobol.yaml`](./category3_parameter_search_sobol.yaml)
- Seed mining (20 videos): [`category1_seed_mining_template.yaml`](./category1_seed_mining_template.yaml)

---

## Three Workflow Types

### Type 1: Base WAN2.2 (No LoRA) - Traditional Sampling
- **Steps**: 15-30 per sampler node
- **Use case**: Highest quality, slower generation
- **Best samplers**: Any work well with more steps
  - `dpmpp_2m`, `dpmpp_2m_sde` (excellent quality)
  - `euler`, `euler_a` (reliable baseline)
  - `heun`, `dpm_2` (high quality, slower)
  - `uni_pc`, `uni_pc_bh2` (good balance)
- **Best schedulers**:
  - `karras` (often best quality)
  - `exponential` (good detail)
  - `normal` (solid baseline)

### Type 2: Light LoRAs (LightX2V) - Fast Sampling ⭐ CURRENT
- **Steps**: 2-4 per sampler node (fixed by LoRA training)
- **Use case**: Fast generation, 4-10x speedup
- **Critical**: LoRAs are trained for specific step counts!
- **Best samplers** (few-step optimized):
  - `euler` (reliable, stable)
  - `euler_a` (adds variation)
  - `dpmpp_2m` (good quality in few steps)
  - `dpmpp_2m_sde` (detail preservation)
- **Avoid**: Samplers requiring many steps
  - `heun`, `dpm_2`, `dpm_2_a` (need 10+ steps)
  - `lms` (unstable at low steps)
- **Best schedulers**:
  - `simple` (LoRAs often trained with this)
  - `normal` (good baseline)
  - `karras` (may improve quality)
- **Note**: Step count should match LoRA training (4 for lightx2v_4steps)

### Type 3: SNR-Based Dynamic Scheduling
- **Steps**: Computed dynamically by SNR scheduler node
- **Use case**: Adaptive quality/speed tradeoff
- **Sampler/Scheduler**: Often controlled by the SNR node
- **Config approach**: Vary SNR node parameters instead of sampler settings

---

## Sampler Compatibility Matrix

### Works Well with 2-4 Steps (Light LoRAs)
| Sampler | Simple | Normal | Karras | Exponential | Notes |
|---------|--------|--------|--------|-------------|-------|
| euler | ✅ | ✅ | ✅ | ✅ | Best all-around |
| euler_a | ✅ | ✅ | ✅ | ⚠️ | Adds randomness |
| dpmpp_2m | ✅ | ✅ | ✅ | ✅ | High quality |
| dpmpp_2m_sde | ✅ | ✅ | ✅ | ✅ | Good detail |
| dpmpp_sde | ⚠️ | ⚠️ | ✅ | ⚠️ | Better with karras |
| ddim | ✅ | ✅ | ❌ | ❌ | Works but deterministic |

### Requires 10+ Steps (Base Model)
| Sampler | Simple | Normal | Karras | Exponential | Notes |
|---------|--------|--------|--------|-------------|-------|
| heun | ✅ | ✅ | ✅ | ✅ | High quality, slow |
| dpm_2 | ✅ | ✅ | ✅ | ✅ | Needs many steps |
| dpm_2_a | ✅ | ✅ | ✅ | ✅ | Ancestral variant |
| uni_pc | ✅ | ✅ | ⚠️ | ✅ | Fast convergence |
| lms | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Unpredictable |

Legend:
- ✅ = Recommended, works well
- ⚠️ = May work, test carefully
- ❌ = Incompatible or poor results

---

## Config Selection Guide

### For Your Current Workflow (Type 2: Light LoRAs)

**Use**: `samplers_type2_lightlora.yaml`
- Fixed 4 steps (matches LoRA training)
- Samplers optimized for few-step generation
- Schedulers that work well with distilled models

**Expected Results**: 4 samplers × 3 schedulers × 2 seeds = **24 videos**

### For Base Model Testing (Type 1)

**Use**: `samplers_type1_base.yaml`
- 20-30 steps for quality
- Full range of samplers
- All schedulers available

**Expected Results**: Much longer generation time per video

### For SNR-Based Workflows (Type 3)

**Use**: Custom config varying SNR node parameters
- Dynamic step computation
- Explore SNR thresholds instead of fixed steps

---

## Recommended Testing Order

1. **Start Small** (Type 2 - Current):
   ```bash
   cr-batch workflows/wan2_14B_flf2v/configs/samplers_type2_lightlora.yaml
   ```
   - Quick results (4 steps)
   - 24 videos total
   - Identifies best sampler/scheduler for your setup

2. **Deep Dive** (Type 1 - Base Model):
   - After finding good combos from step 1
   - Test same combos at 20+ steps
   - Compare quality vs speed tradeoff

3. **Optimize** (Type 3 - SNR):
   - Once you know optimal sampler/scheduler
   - Explore dynamic scheduling
   - Find quality/speed sweet spot

---

## Quality Expectations by Type

### Type 2: Light LoRAs (4 steps)
- **Quality**: 85-95% of base model
- **Speed**: 4-10x faster
- **Best for**: Rapid iteration, testing prompts/scenes
- **Issues to watch**:
  - May have slight temporal jitter
  - Less fine detail than base model
  - Some motion may be simplified

### Type 1: Base Model (20-30 steps)
- **Quality**: Maximum possible
- **Speed**: Baseline (slowest)
- **Best for**: Final production, hero shots
- **Issues to watch**:
  - Very slow iteration
  - High VRAM usage
  - Diminishing returns above 25 steps

### Type 3: SNR-Based
- **Quality**: Adaptive (60-100% depending on SNR threshold)
- **Speed**: Dynamic (faster for simple scenes)
- **Best for**: Production pipeline with variable quality needs
- **Issues to watch**:
  - Harder to predict output quality
  - Requires SNR node tuning

---

## References

- WAN2.2 LightX2V LoRAs are distilled models trained for 4-step generation
- Based on consistency distillation / progressive distillation techniques
- Similar to LCM (Latent Consistency Models) and SDXL Lightning
- Step count is critical - using wrong step count degrades quality significantly

---

**Last Updated**: 2025-11-11
