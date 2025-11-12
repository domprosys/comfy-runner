# ComfyUI Sampler Groups

Organized sampler collections for batch experimentation. Edit these groups as you discover what works best for your workflows.

---

## Group 1: Euler Family
**Philosophy**: Simple first-order methods, fast, predictable

```yaml
samplers:
  - euler
  - euler_cfg_pp          # Euler with Classifier-Free Guidance post-processing
  - euler_ancestral       # Euler with noise injection (stochastic)
  - euler_ancestral_cfg_pp
```

**Best for**:
- Quick tests and baselines
- Low step counts (4-8 steps)
- When you want predictable, repeatable results (non-ancestral) or variety (ancestral)

**Recommended schedulers**: `normal`, `karras`, `exponential`

---

## Group 2: High-Order Predictors
**Philosophy**: More accurate multi-step methods, better quality per step

```yaml
samplers:
  - heun              # 2nd-order Heun's method
  - heunpp2           # Heun++ variant
  - uni_pc            # Unified Predictor-Corrector
  - uni_pc_bh2        # UniPC with BH2 coefficients
  - ipndm             # Improved Pseudo Numerical methods for Diffusion Models
  - ipndm_v           # IPNDM variant
```

**Best for**:
- Medium step counts (15-30 steps)
- When you need accuracy without too many steps
- Base model workflows prioritizing quality

**Recommended schedulers**: `karras`, `exponential`, `beta57`

---

## Group 3: DPM++ Family
**Philosophy**: State-of-the-art DPM solvers, balanced speed/quality

```yaml
samplers:
  - dpmpp_2m          # DPM++ 2M (multistep, deterministic)
  - dpmpp_2m_sde      # DPM++ 2M with stochastic component
  - dpmpp_2m_sde_gpu  # GPU-optimized SDE variant
  - dpmpp_3m_sde      # 3rd-order multistep SDE
  - uni_pc            # Also works great with DPM-style schedules
  - uni_pc_bh2
```

**Best for**:
- All-purpose workflows (4-40 steps)
- When you want the "safe bet" samplers
- Great balance of speed and quality

**Recommended schedulers**: `karras`, `exponential`, `bong_tangent`, `align_your_steps`

---

## Scheduler Reference

Current available schedulers:

### General Purpose
- `normal` - Linear/uniform schedule
- `karras` - Karras et al. schedule (often best quality)
- `exponential` - Exponential decay

### Advanced/Custom
- `beta` - Beta distribution
- `beta57` - Beta with α=5, β=7 (custom tuned)
- `bong_tangent` - Tangent-based schedule
- `align_your_steps` - Optimized from research paper

### Specialized
- `sgm_uniform` - Stability AI's uniform schedule
- `simple` - Simple linear
- `ddim_uniform` - For DDIM sampler
- `lcm` - Latent Consistency Models (LCM-specific only)

---

## Usage with Category Generator

To use these groups in your configs:

### Option A: Manual - Copy sampler list
```yaml
parameters:
  sampler_name:
    type: values
    values: ['euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp']
```

### Option B: Reference group in config (future feature)
```yaml
parameters:
  sampler_name:
    type: sampler_group
    group: euler_family
```

---

## Notes

- **ODE vs SDE**: ODE samplers are deterministic (same seed = same result). SDE adds controlled randomness.
- **Ancestral samplers**: Inject noise at each step, good for variety but less controllable
- **CFG++ variants**: Post-process classifier-free guidance for potentially better quality
- **GPU-optimized**: `_gpu` suffixed samplers are optimized for CUDA

---

## Changelog

- 2025-11-12: Initial groups (Euler, High-Order, DPM++)
