#!/usr/bin/env python3
"""
Batch Parameter Explorer for ComfyUI Workflows
Generates multiple workflow variations with different parameters.
"""

import json
import sys
import csv
import copy
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml
import numpy as np
from scipy.stats import qmc


def load_config(config_path: str) -> Dict[str, Any]:
    """Load batch configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_sobol_samples(param_ranges: Dict[str, Any], n_samples: int) -> List[Dict[str, float]]:
    """
    Generate parameter samples using Sobol quasi-random sequences.
    Provides excellent coverage of parameter space.
    """
    # Separate continuous and discrete parameters
    continuous_params = {}
    discrete_params = {}

    for name, spec in param_ranges.items():
        if isinstance(spec, dict):
            if 'min' in spec and 'max' in spec:
                continuous_params[name] = (spec['min'], spec['max'])
            elif 'values' in spec:
                discrete_params[name] = spec['values']
        else:
            discrete_params[name] = spec  # Assume it's a list

    # Generate Sobol samples for continuous parameters
    samples = []

    if continuous_params:
        dim = len(continuous_params)
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_samples = sampler.random(n_samples)

        # Scale to parameter ranges
        param_names = list(continuous_params.keys())
        for i in range(n_samples):
            sample = {}
            for j, name in enumerate(param_names):
                min_val, max_val = continuous_params[name]
                sample[name] = min_val + sobol_samples[i, j] * (max_val - min_val)

            # Add discrete parameters (randomly for now, or cycle through)
            for name, values in discrete_params.items():
                sample[name] = random.choice(values)

            samples.append(sample)
    else:
        # Only discrete parameters - use grid or random
        for i in range(n_samples):
            sample = {}
            for name, values in discrete_params.items():
                sample[name] = random.choice(values)
            samples.append(sample)

    return samples


def generate_lhs_samples(param_ranges: Dict[str, Any], n_samples: int) -> List[Dict[str, float]]:
    """
    Generate parameter samples using Latin Hypercube Sampling.
    Good for understanding individual parameter effects.
    """
    continuous_params = {}
    discrete_params = {}

    for name, spec in param_ranges.items():
        if isinstance(spec, dict):
            if 'min' in spec and 'max' in spec:
                continuous_params[name] = (spec['min'], spec['max'])
            elif 'values' in spec:
                discrete_params[name] = spec['values']
        else:
            discrete_params[name] = spec

    samples = []

    if continuous_params:
        dim = len(continuous_params)
        sampler = qmc.LatinHypercube(d=dim)
        lhs_samples = sampler.random(n=n_samples)

        param_names = list(continuous_params.keys())
        for i in range(n_samples):
            sample = {}
            for j, name in enumerate(param_names):
                min_val, max_val = continuous_params[name]
                sample[name] = min_val + lhs_samples[i, j] * (max_val - min_val)

            for name, values in discrete_params.items():
                sample[name] = random.choice(values)

            samples.append(sample)
    else:
        for i in range(n_samples):
            sample = {}
            for name, values in discrete_params.items():
                sample[name] = random.choice(values)
            samples.append(sample)

    return samples


def generate_grid_samples(param_ranges: Dict[str, Any]) -> List[Dict[str, float]]:
    """
    Generate all combinations in a grid search.
    Warning: Can be very large!
    """
    # Convert all parameters to discrete values
    param_values = {}

    for name, spec in param_ranges.items():
        if isinstance(spec, dict):
            if 'values' in spec:
                param_values[name] = spec['values']
            elif 'min' in spec and 'max' in spec:
                # For continuous, create reasonable grid (10 points)
                param_values[name] = list(np.linspace(spec['min'], spec['max'], 10))
        else:
            param_values[name] = spec

    # Generate all combinations
    param_names = list(param_values.keys())
    samples = []

    def recursive_grid(index, current_sample):
        if index == len(param_names):
            samples.append(dict(current_sample))
            return

        name = param_names[index]
        for value in param_values[name]:
            current_sample[name] = value
            recursive_grid(index + 1, current_sample)

    recursive_grid(0, {})
    return samples


def generate_random_samples(param_ranges: Dict[str, Any], n_samples: int) -> List[Dict[str, float]]:
    """Generate random parameter samples."""
    samples = []

    for _ in range(n_samples):
        sample = {}
        for name, spec in param_ranges.items():
            if isinstance(spec, dict):
                if 'min' in spec and 'max' in spec:
                    sample[name] = random.uniform(spec['min'], spec['max'])
                elif 'values' in spec:
                    sample[name] = random.choice(spec['values'])
            else:
                sample[name] = random.choice(spec)
        samples.append(sample)

    return samples


def generate_parameter_samples(config: Dict[str, Any]) -> List[Dict[str, float]]:
    """Generate parameter samples based on configuration."""
    strategy = config.get('sampling_strategy', 'sobol')
    param_ranges = config['parameters']
    n_samples = config.get('num_samples', 100)

    print(f"\n📊 Generating parameter samples using {strategy.upper()} strategy...")

    if strategy == 'sobol':
        samples = generate_sobol_samples(param_ranges, n_samples)
    elif strategy == 'lhs':
        samples = generate_lhs_samples(param_ranges, n_samples)
    elif strategy == 'grid':
        samples = generate_grid_samples(param_ranges)
        print(f"   Grid search generated {len(samples)} combinations")
    elif strategy == 'random':
        samples = generate_random_samples(param_ranges, n_samples)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    print(f"   Generated {len(samples)} parameter combinations")
    return samples


def modify_workflow(base_workflow: Dict, params: Dict[str, float]) -> Dict:
    """
    Modify workflow JSON with new parameters.
    Modifies the specific nodes for the FLF2V workflow.
    """
    workflow = copy.deepcopy(base_workflow)

    # Node 57 & 58: KSampler Advanced (CFG, steps)
    if 'cfg' in params:
        workflow['57']['inputs']['cfg'] = params['cfg']
        workflow['58']['inputs']['cfg'] = params['cfg']

    if 'total_steps' in params:
        total = int(params['total_steps'])
        if 'steps_high' in params:
            high = int(params['steps_high'])
        else:
            high = max(1, total // 2)  # Default split

        workflow['57']['inputs']['steps'] = total
        workflow['57']['inputs']['end_at_step'] = high
        workflow['58']['inputs']['steps'] = total
        workflow['58']['inputs']['start_at_step'] = high

    # Node 57: Noise seed
    if 'seed' in params:
        workflow['57']['inputs']['noise_seed'] = int(params['seed'])

    # Node 91 & 92: LoRA strength
    if 'lora_strength' in params:
        workflow['91']['inputs']['strength_model'] = params['lora_strength']
        workflow['92']['inputs']['strength_model'] = params['lora_strength']

    # Node 54 & 55: Model shift
    if 'shift' in params:
        workflow['54']['inputs']['shift'] = int(params['shift'])
        workflow['55']['inputs']['shift'] = int(params['shift'])

    return workflow


def load_progress(progress_file: Path) -> Dict[str, Any]:
    """Load progress tracking data."""
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'failed_runs': []}


def save_progress(progress_file: Path, progress: Dict[str, Any]):
    """Save progress tracking data atomically."""
    temp_file = progress_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(progress, f, indent=2)
    temp_file.replace(progress_file)


def format_run_name(pattern: str, run_id: int, params: Dict[str, float]) -> str:
    """Format run directory name from pattern."""
    format_params = {'run_id': run_id, **params}
    try:
        return pattern.format(**format_params)
    except KeyError as e:
        print(f"⚠️  Warning: Pattern variable {e} not in parameters, using run_id only")
        return f"{run_id:04d}"


def run_workflow_simple(workflow: Dict, server_address: str, poll_interval: int) -> Tuple[bool, str, List[Path]]:
    """
    Simplified workflow runner (extracted from comfy_runner.py).
    Returns: (success, prompt_id, output_files)
    """
    import time
    import urllib.request
    import urllib.parse

    def queue_prompt(prompt):
        payload = {"prompt": prompt}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result['prompt_id']

    def get_history(prompt_id):
        url = f"http://{server_address}/history/{prompt_id}"
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

    try:
        # Submit workflow
        prompt_id = queue_prompt(workflow)

        # Poll until complete
        while not is_execution_complete(prompt_id):
            time.sleep(poll_interval)

        # Get outputs metadata (don't download yet)
        history = get_history(prompt_id)
        if prompt_id not in history:
            return False, prompt_id, []

        return True, prompt_id, []

    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False, "", []


def download_outputs(prompt_id: str, output_dir: Path, server_address: str) -> List[Path]:
    """Download output files from ComfyUI."""
    import urllib.request
    import urllib.parse

    def get_history(prompt_id):
        url = f"http://{server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def download_file_streaming(filename, subfolder, folder_type, save_path):
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        url_params = urllib.parse.urlencode(params)
        url = f"http://{server_address}/view?{url_params}"

        with urllib.request.urlopen(url) as response:
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)

    history = get_history(prompt_id)
    outputs = history[prompt_id].get('outputs', {})
    saved_files = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for node_id, node_output in outputs.items():
        for media_key in ['images', 'gifs', 'videos']:
            if media_key in node_output:
                for media in node_output[media_key]:
                    filename = media['filename']
                    subfolder = media.get('subfolder', '')
                    file_type = media.get('type', 'output')

                    save_path = output_dir / filename
                    download_file_streaming(filename, subfolder, file_type, save_path)
                    saved_files.append(save_path)

    return saved_files


def run_batch(config_path: str):
    """Main batch execution function."""
    # Load configuration
    print("=" * 70)
    print("🎬 ComfyUI Batch Parameter Explorer")
    print("=" * 70)

    config = load_config(config_path)
    print(f"\n✓ Loaded config: {config_path}")

    # Load base workflow
    workflow_file = Path(config['workflow_file'])
    if not workflow_file.exists():
        print(f"❌ Error: Workflow file not found: {workflow_file}")
        sys.exit(1)

    with open(workflow_file, 'r') as f:
        base_workflow = json.load(f)
    print(f"✓ Loaded workflow: {workflow_file}")

    # Generate parameter samples
    param_samples = generate_parameter_samples(config)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = Path(config['output']['base_dir']) / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    print(f"\n📁 Output directory: {batch_dir}")

    # Save config copy
    config_copy = batch_dir / "config.yaml"
    with open(config_copy, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup progress tracking
    progress_file = batch_dir / "progress.json"
    progress = load_progress(progress_file)

    # Setup metadata CSV
    metadata_file = batch_dir / "metadata.csv"
    csv_file = open(metadata_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'run_id', 'lora_strength', 'cfg', 'steps_high', 'total_steps',
        'shift', 'seed', 'prompt_id', 'output_path', 'status', 'timestamp'
    ])
    csv_writer.writeheader()

    # Generate all runs (param samples × seeds)
    seeds_per_sample = config.get('seeds_per_sample', 1)
    total_runs = len(param_samples) * seeds_per_sample

    print(f"\n🚀 Starting batch execution:")
    print(f"   Parameter combinations: {len(param_samples)}")
    print(f"   Seeds per combination: {seeds_per_sample}")
    print(f"   Total runs: {total_runs}")
    print(f"   Resume enabled: {config.get('resume', {}).get('enabled', False)}")
    print("=" * 70)

    server_address = config['comfyui']['server_address']
    poll_interval = config['comfyui']['poll_interval']
    run_name_pattern = config['output'].get('run_name_pattern', '{run_id:04d}')

    run_id = 0
    completed = 0
    failed = 0

    for sample_idx, params in enumerate(param_samples):
        for seed_idx in range(seeds_per_sample):
            run_id += 1

            # Generate unique seed
            seed = random.randint(0, 2**32 - 1)
            run_params = {**params, 'seed': seed}

            # Format run directory name
            run_name = format_run_name(run_name_pattern, run_id, run_params)
            run_dir = runs_dir / run_name

            # Check if already completed (resume)
            if run_name in progress['completed_runs']:
                print(f"\n[{run_id}/{total_runs}] ⏭️  Skipping (already completed): {run_name}")
                completed += 1
                continue

            print(f"\n[{run_id}/{total_runs}] ▶️  Running: {run_name}")
            print(f"   Params: {run_params}")

            # Modify workflow
            workflow = modify_workflow(base_workflow, run_params)

            # Run workflow
            success, prompt_id, _ = run_workflow_simple(workflow, server_address, poll_interval)

            if success:
                print(f"   ✓ Execution complete (prompt_id: {prompt_id})")

                # Download outputs
                print(f"   📥 Downloading outputs...")
                output_files = download_outputs(prompt_id, run_dir, server_address)

                if output_files:
                    print(f"   ✓ Saved {len(output_files)} file(s) to: {run_dir}")

                    # Save parameters JSON
                    if config['output'].get('save_params_json', True):
                        params_file = run_dir / "params.json"
                        with open(params_file, 'w') as f:
                            json.dump(run_params, f, indent=2)

                    # Update progress
                    progress['completed_runs'].append(run_name)
                    save_progress(progress_file, progress)

                    # Write metadata
                    csv_writer.writerow({
                        'run_id': run_id,
                        'lora_strength': run_params.get('lora_strength', ''),
                        'cfg': run_params.get('cfg', ''),
                        'steps_high': run_params.get('steps_high', ''),
                        'total_steps': run_params.get('total_steps', ''),
                        'shift': run_params.get('shift', ''),
                        'seed': run_params.get('seed', ''),
                        'prompt_id': prompt_id,
                        'output_path': str(run_dir),
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                    csv_file.flush()

                    completed += 1
                else:
                    print(f"   ⚠️  No outputs generated")
                    failed += 1
                    progress['failed_runs'].append(run_name)
                    save_progress(progress_file, progress)
            else:
                print(f"   ❌ Execution failed")
                failed += 1
                progress['failed_runs'].append(run_name)
                save_progress(progress_file, progress)

                csv_writer.writerow({
                    'run_id': run_id,
                    **{k: v for k, v in run_params.items()},
                    'prompt_id': prompt_id or 'N/A',
                    'output_path': str(run_dir),
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat()
                })
                csv_file.flush()

    csv_file.close()

    # Summary
    print("\n" + "=" * 70)
    print("🎉 Batch execution complete!")
    print("=" * 70)
    print(f"   Total runs: {total_runs}")
    print(f"   Completed: {completed}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {completed/total_runs*100:.1f}%")
    print(f"\n📁 Results saved to: {batch_dir}")
    print(f"📊 Metadata CSV: {metadata_file}")
    print("\n💡 Next steps:")
    print(f"   python generate_contact_sheet.py {batch_dir}")
    print("=" * 70)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "batch_config.yaml"
    run_batch(config_file)
