# RES4LYF Package Analysis

## Overview

RES4LYF (Restart Samplers 4 Life) is an advanced custom node package for ComfyUI created by ClownsharkBatwing that implements **Restart Samplers** - optimized numerical solvers for diffusion model denoising.

**GitHub**: https://github.com/ClownsharkBatwing/RES4LYF

---

## Key Innovation: Restart Samplers

**The Problem**: Traditional samplers (euler, dpmpp_2m) require 20-30+ steps for good quality.

**The Solution**: Restart samplers use higher-order Runge-Kutta methods with optimized coefficients for **specific step counts**.

### Why "Restart"?

Restart samplers are designed with coefficients specifically tuned for 2, 3, 5, or 6 steps. They "restart" the integration at each step with optimized phi functions.

---

## Sampler Naming Convention

### RES Family (Restart Samplers)

| Sampler | Type | Steps | ODE/SDE | Description |
|---------|------|-------|---------|-------------|
| `res_2m` | Multistep | 2 | SDE | 2-step multistep, stochastic |
| `res_2m_ode` | Multistep | 2 | ODE | 2-step multistep, deterministic |
| `res_3m` | Multistep | 3 | SDE | 3-step multistep, stochastic |
| `res_3m_ode` | Multistep | 3 | ODE | 3-step multistep, deterministic |
| `res_2s` | Singlestep | 2 | SDE | 2-step singlestep, stochastic |
| `res_2s_ode` | Singlestep | 2 | ODE | 2-step singlestep, deterministic |
| `res_3s` | Singlestep | 3 | SDE | 3-step singlestep, stochastic |
| `res_3s_ode` | Singlestep | 3 | ODE | 3-step singlestep, deterministic |
| `res_5s` | Singlestep | 5 | SDE | 5-step singlestep, stochastic |
| `res_5s_ode` | Singlestep | 5 | ODE | 5-step singlestep, deterministic |
| `res_6s` | Singlestep | 6 | SDE | 6-step singlestep, stochastic |
| `res_6s_ode` | Singlestep | 6 | ODE | 6-step singlestep, deterministic |

**Multistep vs Singlestep**:
- **Multistep**: Uses history from previous steps (more accurate, slightly complex)
- **Singlestep**: Independent steps (simpler, robust)

**ODE vs SDE**:
- **ODE** (`_ode` suffix): `eta=0.0` - Deterministic, same seed = same output
- **SDE** (no suffix): `eta > 0.0` - Stochastic noise injection, more variation

### DEIS Family (Diffusion Exponential Integrator Sampler)

| Sampler | Type | ODE/SDE | Description |
|---------|------|---------|-------------|
| `deis_2m` | 2nd order multistep | SDE | Exponential integrator, 2nd order |
| `deis_2m_ode` | 2nd order multistep | ODE | Deterministic variant |
| `deis_3m` | 3rd order multistep | SDE | Exponential integrator, 3rd order |
| `deis_3m_ode` | 3rd order multistep | ODE | Deterministic variant |

**DEIS Theory**: Uses exponential integrators from the differential equations literature. More mathematically sophisticated than standard Runge-Kutta.

---

## Code Architecture

### How Samplers are Registered

From `/beta/__init__.py` lines 90-109:

```python
extra_samplers.update({
    "res_2m"     : sample_res_2m,
    "res_3m"     : sample_res_3m,
    # ... all call rk_sampler_beta.sample_rk_beta with different rk_type
})
```

All RES samplers funnel through a single master function: `rk_sampler_beta.sample_rk_beta()` with different `rk_type` parameters.

### Key Files

| File | Purpose |
|------|---------|
| `beta/rk_sampler_beta.py` | Main sampler implementation (149KB!) |
| `beta/rk_coefficients_beta.py` | Runge-Kutta coefficients for each sampler type (128KB) |
| `beta/rk_guide_func_beta.py` | Guidance functions, CFG++, detail boost (134KB) |
| `beta/rk_noise_sampler_beta.py` | Noise injection for SDE variants (42KB) |
| `beta/deis_coefficients.py` | DEIS-specific coefficients (6KB) |
| `beta/samplers_extensions.py` | Node classes for options (217KB!) |
| `sigmas.py` | Scheduler implementations (157KB) |

---

## New Scheduler: `bong_tangent`

From `__init__.py` lines 20-25:

```python
from comfy.samplers import SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
new_scheduler_name = "bong_tangent"
if new_scheduler_name not in SCHEDULER_HANDLERS:
    bong_tangent_handler = SchedulerHandler(handler=sigmas.bong_tangent_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[new_scheduler_name] = bong_tangent_handler
    SCHEDULER_NAMES.append(new_scheduler_name)
```

**`bong_tangent`**: A custom scheduler using tangent-based noise schedule (likely non-linear decay).

---

## Advanced Features (Beyond Basic Sampling)

RES4LYF is not just samplers - it's a full sampling framework with:

### 1. **CFG++ Variants**
Samplers with `_cfg_pp` suffix use **Classifier-Free Guidance post-processing**:
- `euler_cfg_pp`
- `euler_ancestral_cfg_pp`
- `dpmpp_2m_cfg_pp`
- `res_multistep_cfg_pp`

**What it does**: Improves CFG behavior by post-processing the guidance vector.

### 2. **Detail Boost**
From `samplers_extensions.py` lines 161-200:

```python
@dataclass
class DetailBoostOptions:
    noise_scaling_weight : float = 0.0  # Positive = sharper, negative = softer
    noise_boost_step     : float = 0.0
    noise_boost_substep  : float = 0.0
```

Allows you to **sharpen** (positive) or **soften** (negative) images during sampling.

### 3. **SDE Options**
From lines 49-117:

```python
class ClownOptions_SDE_Beta:
    noise_type_sde:         # gaussian, perlin, fractal, etc.
    noise_mode_sde:         # hard, soft, sinusoidal, etc.
    eta:                    # Noise injection amount (0 = ODE, >0 = SDE)
    eta_substep:            # For substeps
```

Fine control over stochastic noise injection.

### 4. **Style Transfer**
- `ClownGuide_Style_Beta`
- `VAEStyleTransferLatent`

Transfer style from reference images during generation.

### 5. **Regional Conditioning**
- `ClownRegionalConditioning`
- `TemporalMaskGenerator` (for video!)

Apply different prompts to different regions or frames.

---

## For WAN2.2 Video Generation

### Relevant Nodes

1. **`ClownpileModelWanVideo`** (models.py:189)
   - Torch compile optimization for WAN video models

2. **`ReWanPatcher`** (models.py:197)
   - Model patching for WAN-specific features

3. **`ReWanPatcherAdvanced`** (models.py:207)
   - Advanced WAN patching

4. **Temporal Features**:
   - `TemporalMaskGenerator` - Masks for video frames
   - `TemporalSplitAttnMask` - Split attention across time
   - `TemporalCrossAttnMask` - Cross-frame attention

---

## Practical Recommendations for 4-Step Light LoRA

### Best RES Samplers for 4 Steps

Since your LoRAs are trained for 4 steps:

**Top Choices**:
1. `res_3m` or `res_3m_ode` - Closest to 4 steps, multistep
2. `res_3s` or `res_3s_ode` - 3-step singlestep (simpler)
3. `res_5s` or `res_5s_ode` - Slightly above, might work

**Not Recommended**:
- `res_2m`, `res_2s` - Too few steps (2)
- `res_6s` - Too many steps (6)

### ODE vs SDE for Testing

**For systematic exploration** (finding best sampler/scheduler combo):
- Use **ODE variants** (`res_3m_ode`, `res_3s_ode`)
- Deterministic = easier to compare results
- Same seed = same output = fair comparison

**For creative variation**:
- Use **SDE variants** (`res_3m`, `res_3s`)
- Each seed gives different results
- More diversity

---

## Implementation Details

### How `_ode` Works

From `beta/__init__.py` lines 175-186:

```python
def sample_res_3m_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return rk_sampler_beta.sample_rk_beta(
        model, x, sigmas, None, extra_args, callback, disable,
        rk_type="res_3m",
        eta=0.0,          # ← Zero noise = ODE
        eta_substep=0.0,  # ← Zero substep noise = ODE
    )
```

**ODE = SDE with eta set to 0!**

### Runge-Kutta Type Codes

From the code, `rk_type` values:
- `"res_2m"`, `"res_3m"` - RES multistep
- `"res_2s"`, `"res_3s"`, `"res_5s"`, `"res_6s"` - RES singlestep
- `"deis_2m"`, `"deis_3m"` - DEIS exponential integrators
- Plus many more in `rk_coefficients_beta.py`

---

## Missing: res4lyf_looper

**Note**: You mentioned `res4lyf_looper` but I couldn't find it installed.

Possible options:
1. Not yet installed
2. Part of a different package
3. Deprecated/renamed feature
4. Refers to the "Chainsampler" nodes:
   - `SharkChainsampler_Beta`
   - `ClownsharkChainsampler_Beta`

**Chainsamplers**: Run multiple samplers sequentially in one workflow (mentioned in README line 51).

---

## Next Steps for Batch Testing

### Updated Sampler List for Type 2 (4-step Light LoRA)

**Add these RES samplers**:
```yaml
sampler_name:
  path: nodes.57.inputs.sampler_name
  type: values
  values:
    # Original
    - "euler"
    - "euler_a"
    - "dpmpp_2m"
    - "dpmpp_2m_sde"

    # RES4LYF additions (3-step optimized)
    - "res_3m_ode"      # Deterministic 3-step multistep
    - "res_3s_ode"      # Deterministic 3-step singlestep
    - "res_5s_ode"      # Deterministic 5-step (might work at 4)

    # CFG++ variants
    - "euler_cfg_pp"
    - "dpmpp_2m_cfg_pp"

    # DEIS (exponential integrators)
    - "deis_2m_ode"
    - "deis_3m_ode"
```

**Add new scheduler**:
```yaml
scheduler:
  path: nodes.57.inputs.scheduler
  type: values
  values:
    - "normal"
    - "karras"
    - "exponential"
    - "bong_tangent"    # New RES4LYF scheduler
```

---

## References

- **RES4LYF README**: Claims RES_3M achieves better quality in 20 steps than Uni-PC in 50+ steps
- **Workflow examples**: `/example_workflows/intro to clownsampling.json`
- **Community**: ClownsharkBatwing (author) active on GitHub

---

**Status**: Installed and ready to test
**Version**: Beta (active development)
**Compatibility**: WAN, Flux, SD3.5, HiDream, AuraFlow, SDXL, SD1.5, Stable Cascade, LTXV

**Last Updated**: 2025-11-11
