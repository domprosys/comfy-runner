# Type 3: SNR-Based Dynamic Scheduling

## Overview

SNR (Signal-to-Noise Ratio) schedulers dynamically compute the number of sampling steps based on:
- The complexity of the scene
- Desired quality level
- Time budget constraints

This is different from fixed-step workflows (Types 1 & 2).

## How SNR Schedulers Work

Instead of fixed steps like `steps: 20`, an SNR scheduler node computes:
- **When to stop**: Based on SNR threshold (e.g., stop when SNR < 0.1)
- **Step size**: Adaptive based on noise level
- **CFG/Shift**: May also be computed dynamically

## ComfyUI SNR Scheduler Nodes

Common SNR scheduler implementations:
- **AlignYourStepsScheduler** (AYS)
- **Gits Scheduler**
- **Custom SNR nodes** from community

## Typical Workflow Structure

```
Load Model
    ↓
SNR Scheduler Node
  ├─ Input: target_snr, max_steps, model_type
  ├─ Output: sigmas (noise schedule)
    ↓
KSampler
  ├─ Uses dynamic sigmas instead of fixed steps
  ├─ Sampler/scheduler may be overridden by SNR node
    ↓
Output
```

## Parameters to Explore

Instead of varying `sampler_name` and `scheduler`, you'd vary:

1. **SNR Threshold** (quality vs speed)
   ```yaml
   snr_threshold:
     path: nodes.XX.inputs.target_snr
     type: linear
     min: 0.05  # Lower = more steps = higher quality
     max: 0.5   # Higher = fewer steps = faster
     step: 0.05
   ```

2. **Max Steps** (safety limit)
   ```yaml
   max_steps:
     path: nodes.XX.inputs.max_steps
     type: values
     values: [15, 25, 35, 50]
   ```

3. **Model Type** (affects SNR calculation)
   ```yaml
   model_type:
     path: nodes.XX.inputs.model_type
     type: values
     values: ["SD3", "FLUX", "WAN"]
   ```

4. **Denoise Strength** (for img2img/video)
   ```yaml
   denoise:
     path: nodes.XX.inputs.denoise
     type: linear
     min: 0.7
     max: 1.0
     step: 0.05
   ```

## Advantages

- **Adaptive Quality**: Simple scenes finish faster
- **Consistent Quality**: Complex scenes get more steps automatically
- **Production Pipelines**: Different quality tiers without manual tuning

## Disadvantages

- **Less Predictable**: Output time varies
- **Complex Setup**: Requires understanding SNR theory
- **Harder to Compare**: Each run may use different step counts

## Creating a Type 3 Config

1. **Identify your SNR scheduler node** in the workflow
   ```bash
   python -c "
   import json
   with open('workflow.json') as f:
       wf = json.load(f)
   for nid, node in wf.items():
       if 'SNR' in node.get('class_type', '') or 'AlignYourSteps' in node.get('class_type', ''):
           print(f'Node {nid}: {node.get(\"class_type\")}')
           print(f'  Inputs: {list(node.get(\"inputs\", {}).keys())}')
   "
   ```

2. **Create config based on node inputs**
   ```yaml
   workflow_file: ../workflow.json
   sampling_strategy: sobol
   num_samples: 50

   parameters:
     snr_threshold:
       path: nodes.XX.inputs.target_snr
       type: continuous
       min: 0.05
       max: 0.5

     max_steps:
       path: nodes.XX.inputs.max_steps
       type: linear
       min: 15
       max: 50
       step: 5
   ```

3. **Analyze results by actual steps used**
   - The `metadata.csv` will show varied step counts
   - Compare quality vs actual steps taken
   - Find optimal SNR threshold for your use case

## When to Use Type 3

- **Production pipelines** with mixed complexity scenes
- **Quality tiers** (draft/preview/final)
- **Budget constraints** (maximize quality for time budget)
- **After** you've established baseline quality with Types 1 & 2

## Recommended Testing Order

1. Test Type 2 (Light LoRA) - fast baseline
2. Test Type 1 (Base Model) - quality ceiling
3. **Then** implement Type 3 - adaptive solution between them

---

**Status**: No SNR scheduler nodes detected in current workflow
**To Implement**: Add AlignYourSteps or similar node to workflow first

**Last Updated**: 2025-11-11
