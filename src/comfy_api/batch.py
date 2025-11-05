#!/usr/bin/env python3
"""
Generic Batch Parameter Explorer for ComfyUI Workflows (packaged)
Works with any workflow using parameter paths.
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def expand_parameter_specs(param_specs: Dict[str, Any]) -> Dict[str, Dict]:
    expanded = {}
    for param_name, spec in param_specs.items():
        if not isinstance(spec, dict):
            expanded[param_name] = {'type': 'values', 'values': spec, 'path': param_name}
            continue
        param_type = spec.get('type', 'auto')
        path = spec.get('path', param_name)
        if param_type == 'random_seed':
            expanded[param_name] = {'type': 'random_seed', 'path': path}
        elif param_type == 'linear':
            min_val = spec['min']; max_val = spec['max']; step = spec.get('step', 1)
            values = list(np.arange(min_val, max_val + step/2, step))
            expanded[param_name] = {'type': 'values', 'values': values, 'path': path}
        elif param_type == 'continuous' or ('min' in spec and 'max' in spec and 'step' not in spec):
            expanded[param_name] = {'type': 'continuous', 'min': spec['min'], 'max': spec['max'], 'path': path}
        elif 'values' in spec:
            expanded[param_name] = {'type': 'values', 'values': spec['values'], 'path': path}
        elif param_type == 'auto':
            if 'min' in spec and 'max' in spec:
                if 'step' in spec:
                    min_val = spec['min']; max_val = spec['max']; step = spec['step']
                    values = list(np.arange(min_val, max_val + step/2, step))
                    expanded[param_name] = {'type': 'values', 'values': values, 'path': path}
                else:
                    expanded[param_name] = {'type': 'continuous', 'min': spec['min'], 'max': spec['max'], 'path': path}
    return expanded


def generate_sobol_samples(param_specs: Dict[str, Dict], n_samples: int) -> List[Dict[str, Any]]:
    continuous_params = {}; discrete_params = {}
    for name, spec in param_specs.items():
        if spec['type'] == 'continuous':
            continuous_params[name] = (spec['min'], spec['max'])
        elif spec['type'] == 'values':
            discrete_params[name] = spec['values']
    samples = []
    if continuous_params:
        dim = len(continuous_params)
        sampler = qmc.Sobol(d=dim, scramble=True)
        sobol_samples = sampler.random(n_samples)
        param_names = list(continuous_params.keys())
        for i in range(n_samples):
            sample = {}
            for j, name in enumerate(param_names):
                min_val, max_val = continuous_params[name]
                sample[name] = min_val + sobol_samples[i, j] * (max_val - min_val)
            for name, values in discrete_params.items():
                sample[name] = random.choice(values)
            samples.append(sample)
    else:
        for _ in range(n_samples):
            sample = {name: random.choice(values) for name, values in discrete_params.items()}
            samples.append(sample)
    return samples


def generate_lhs_samples(param_specs: Dict[str, Dict], n_samples: int) -> List[Dict[str, Any]]:
    continuous_params = {}; discrete_params = {}
    for name, spec in param_specs.items():
        if spec['type'] == 'continuous':
            continuous_params[name] = (spec['min'], spec['max'])
        elif spec['type'] == 'values':
            discrete_params[name] = spec['values']
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
        for _ in range(n_samples):
            sample = {name: random.choice(values) for name, values in discrete_params.items()}
            samples.append(sample)
    return samples


def generate_grid_samples(param_specs: Dict[str, Dict]) -> List[Dict[str, Any]]:
    param_values = {}
    for name, spec in param_specs.items():
        if spec['type'] == 'continuous':
            param_values[name] = list(np.linspace(spec['min'], spec['max'], 10))
        elif spec['type'] == 'values':
            param_values[name] = spec['values']
    param_names = list(param_values.keys())
    samples = []
    def recursive_grid(index, current_sample):
        if index == len(param_names):
            samples.append(dict(current_sample)); return
        name = param_names[index]
        for value in param_values[name]:
            current_sample[name] = value
            recursive_grid(index + 1, current_sample)
    recursive_grid(0, {})
    return samples


def generate_random_samples(param_specs: Dict[str, Dict], n_samples: int) -> List[Dict[str, Any]]:
    samples = []
    for _ in range(n_samples):
        sample = {}
        for name, spec in param_specs.items():
            if spec['type'] == 'continuous':
                sample[name] = random.uniform(spec['min'], spec['max'])
            elif spec['type'] == 'values':
                sample[name] = random.choice(spec['values'])
        samples.append(sample)
    return samples


def generate_parameter_samples(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
    strategy = config.get('sampling_strategy', 'sobol')
    param_configs = config.get('parameters', {})
    n_samples = config.get('num_samples', 100)
    print(f"\n📊 Generating parameter samples using {strategy.upper()} strategy...")
    param_specs = expand_parameter_specs(param_configs)
    sampling_specs = {k: v for k, v in param_specs.items() if v['type'] != 'random_seed'}
    if not sampling_specs:
        print("   ⚠️  No parameters to sample (only random seeds?)")
        return [{}], param_specs
    if strategy == 'sobol':
        samples = generate_sobol_samples(sampling_specs, n_samples)
    elif strategy == 'lhs':
        samples = generate_lhs_samples(sampling_specs, n_samples)
    elif strategy == 'grid':
        samples = generate_grid_samples(sampling_specs)
        print(f"   Grid search generated {len(samples)} combinations")
    elif strategy == 'random':
        samples = generate_random_samples(sampling_specs, n_samples)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    print(f"   Generated {len(samples)} parameter combinations")
    return samples, param_specs


def set_nested_value(obj: Dict, path: str, value: Any):
    parts = path.split('.')
    if parts[0] == 'nodes':
        parts = parts[1:]
    current = obj
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def modify_workflow_generic(base_workflow: Dict, params: Dict[str, Any], param_specs: Dict[str, Dict]) -> Dict:
    workflow = copy.deepcopy(base_workflow)
    for param_name, value in params.items():
        if param_name not in param_specs:
            continue
        spec = param_specs[param_name]
        path = spec.get('path', param_name)
        set_nested_value(workflow, path, value)
    return workflow


def load_progress(progress_file: Path) -> Dict[str, Any]:
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'failed_runs': []}


def save_progress(progress_file: Path, progress: Dict[str, Any]):
    temp_file = progress_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(progress, f, indent=2)
    temp_file.replace(progress_file)


def format_run_name(pattern: str, run_id: int, params: Dict[str, Any]) -> str:
    format_params = {'run_id': run_id}
    for key, value in params.items():
        clean_key = key.replace('param_nodes.', '').replace('.inputs.', '_').replace('.', '_')
        if isinstance(value, float):
            format_params[clean_key] = f"{value:.2f}"
        else:
            format_params[clean_key] = str(value)
    try:
        return pattern.format(**format_params)
    except (KeyError, ValueError):
        return f"{run_id:04d}"


def run_workflow_simple(workflow: Dict, server_address: str, poll_interval: int) -> Tuple[bool, str, List[Path]]:
    import time
    import urllib.request
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
        prompt_id = queue_prompt(workflow)
        while not is_execution_complete(prompt_id):
            time.sleep(poll_interval)
        history = get_history(prompt_id)
        if prompt_id not in history:
            return False, prompt_id, []
        return True, prompt_id, []
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False, "", []


def download_outputs(prompt_id: str, output_dir: Path, server_address: str) -> List[Path]:
    import urllib.request
    import urllib.parse
    def get_history(prompt_id):
        url = f"http://{server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())
    def download_file_streaming(filename, subfolder, folder_type, save_path):
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
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
    for _, node_output in outputs.items():
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
    print("=" * 70)
    print("🎬 ComfyUI Generic Batch Parameter Explorer")
    print("=" * 70)
    config = load_config(config_path)
    print(f"\n✓ Loaded config: {config_path}")

    config_dir = Path(config_path).resolve().parent

    workflow_file = Path(config['workflow_file'])
    if not workflow_file.is_absolute():
        workflow_file = (config_dir / workflow_file).resolve()
    if not workflow_file.exists():
        print(f"❌ Error: Workflow file not found: {workflow_file}")
        sys.exit(1)
    with open(workflow_file, 'r') as f:
        base_workflow = json.load(f)
    print(f"✓ Loaded workflow: {workflow_file}")

    param_samples, param_specs = generate_parameter_samples(config)

    output_base = Path(config['output']['base_dir'])
    if not output_base.is_absolute():
        output_base = (config_dir / output_base).resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = output_base / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = batch_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    print(f"\n📁 Output directory: {batch_dir}")

    with open(batch_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    progress_file = batch_dir / "progress.json"
    progress = load_progress(progress_file)

    metadata_file = batch_dir / "metadata.csv"
    param_names = list(param_specs.keys())
    csv_fieldnames = ['run_id'] + param_names + ['prompt_id', 'output_path', 'status', 'timestamp']
    csv_file = open(metadata_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    seeds_per_sample = config.get('seeds_per_sample', 1)
    seed_params = [k for k, v in param_specs.items() if v['type'] == 'random_seed']
    total_runs = len(param_samples) * seeds_per_sample

    print(f"\n🚀 Starting batch execution:")
    print(f"   Parameter combinations: {len(param_samples)}")
    print(f"   Seeds per combination: {seeds_per_sample}")
    print(f"   Random seed parameters: {len(seed_params)}")
    print(f"   Total runs: {total_runs}")
    print("=" * 70)

    server_address = config['comfyui']['server_address']
    poll_interval = config['comfyui']['poll_interval']
    run_name_pattern = config['output'].get('run_name_pattern', '{run_id:04d}')

    run_id = 0
    completed = 0
    failed = 0

    for params in param_samples:
        for _ in range(seeds_per_sample):
            run_id += 1
            run_params = dict(params)
            for seed_param in seed_params:
                run_params[seed_param] = random.randint(0, 2**32 - 1)
            run_name = format_run_name(run_name_pattern, run_id, run_params)
            run_dir = runs_dir / run_name
            if run_name in progress['completed_runs']:
                print(f"\n[{run_id}/{total_runs}] ⏭️  Skipping (already completed): {run_name}")
                completed += 1
                continue
            print(f"\n[{run_id}/{total_runs}] ▶️  Running: {run_name}")
            print(f"   Params: {run_params}")
            workflow = modify_workflow_generic(base_workflow, run_params, param_specs)
            success, prompt_id, _ = run_workflow_simple(workflow, server_address, poll_interval)
            if success:
                print(f"   ✓ Execution complete (prompt_id: {prompt_id})")
                print(f"   📥 Downloading outputs...")
                output_files = download_outputs(prompt_id, run_dir, server_address)
                if output_files:
                    print(f"   ✓ Saved {len(output_files)} file(s)")
                    if config['output'].get('save_params_json', True):
                        with open(run_dir / "params.json", 'w') as f:
                            json.dump(run_params, f, indent=2)
                    progress['completed_runs'].append(run_name)
                    save_progress(progress_file, progress)
                    row = {'run_id': run_id, 'prompt_id': prompt_id,
                           'output_path': str(run_dir), 'status': 'success',
                           'timestamp': datetime.now().isoformat()}
                    for pname in param_names:
                        row[pname] = run_params.get(pname, '')
                    csv_writer.writerow(row)
                    csv_file.flush()
                    completed += 1
                else:
                    print(f"   ⚠️  No outputs generated")
                    failed += 1
            else:
                print(f"   ❌ Execution failed")
                failed += 1
                progress['failed_runs'].append(run_name)
                save_progress(progress_file, progress)

    csv_file.close()

    print("\n" + "=" * 70)
    print("🎉 Batch execution complete!")
    print("=" * 70)
    print(f"   Total runs: {total_runs}")
    print(f"   Completed: {completed}")
    print(f"   Failed: {failed}")
    if total_runs > 0:
        print(f"   Success rate: {completed/total_runs*100:.1f}%")
    print(f"\n📁 Results: {metadata_file.parent}")
    print(f"📊 Metadata: {metadata_file}")
    print("\n💡 Next: python -m comfy_api.contact_sheet {batch_dir}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m comfy_api.batch <config.yaml>")
        sys.exit(1)
    run_batch(sys.argv[1])


if __name__ == "__main__":
    main()

