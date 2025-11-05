#!/usr/bin/env python3
"""
Thin wrapper around the packaged shot-based batch runner.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comfy_api.shots_batch import main


# Import functions from batch_runner_v2
# (In production, would refactor to shared module, but keeping standalone for now)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load batch configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def expand_parameter_specs(param_specs: Dict[str, Any]) -> Dict[str, Dict]:
    """Expand parameter specifications to uniform format."""
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
            min_val, max_val, step = spec['min'], spec['max'], spec.get('step', 1)
            values = list(np.arange(min_val, max_val + step/2, step))
            expanded[param_name] = {'type': 'values', 'values': values, 'path': path}
        elif param_type == 'continuous' or ('min' in spec and 'max' in spec and 'step' not in spec):
            expanded[param_name] = {'type': 'continuous', 'min': spec['min'], 'max': spec['max'], 'path': path}
        elif 'values' in spec:
            expanded[param_name] = {'type': 'values', 'values': spec['values'], 'path': path}

    return expanded


def generate_sobol_samples(param_specs: Dict[str, Dict], n_samples: int) -> List[Dict[str, Any]]:
    """Generate parameter samples using Sobol sequences."""
    continuous_params = {k: (v['min'], v['max']) for k, v in param_specs.items() if v['type'] == 'continuous'}
    discrete_params = {k: v['values'] for k, v in param_specs.items() if v['type'] == 'values'}

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
        for i in range(n_samples):
            sample = {name: random.choice(values) for name, values in discrete_params.items()}
            samples.append(sample)

    return samples


def generate_parameter_samples(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
    """Generate parameter samples based on configuration."""
    strategy = config.get('sampling_strategy', 'sobol')
    param_configs = config.get('parameters', {})
    n_samples = config.get('num_samples', 100)

    param_specs = expand_parameter_specs(param_configs)
    sampling_specs = {k: v for k, v in param_specs.items() if v['type'] != 'random_seed'}

    if not sampling_specs:
        return [{}], param_specs

    if strategy == 'sobol':
        samples = generate_sobol_samples(sampling_specs, n_samples)
    else:
        # Simplified - only sobol for now
        samples = generate_sobol_samples(sampling_specs, n_samples)

    return samples, param_specs


def set_nested_value(obj: Dict, path: str, value: Any):
    """Set a value in a nested dictionary using dot notation path."""
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
    """Modify workflow JSON with new parameters using paths."""
    workflow = copy.deepcopy(base_workflow)
    for param_name, value in params.items():
        if param_name in param_specs:
            path = param_specs[param_name].get('path', param_name)
            set_nested_value(workflow, path, value)
    return workflow


# ========================================
# SHOT-SPECIFIC FUNCTIONS
# ========================================

def discover_shots(shots_dir: Path) -> List[Dict[str, Any]]:
    """
    Discover and validate shot folders.
    Each shot folder must contain: prompt.txt, first_frame.png, last_frame.png
    """
    if not shots_dir.exists():
        print(f"❌ Shots directory not found: {shots_dir}")
        return []

    shots = []
    for shot_dir in sorted(shots_dir.iterdir()):
        if not shot_dir.is_dir():
            continue

        # Check required files
        prompt_file = shot_dir / "prompt.txt"
        first_frame = shot_dir / "first_frame.png"
        last_frame = shot_dir / "last_frame.png"

        missing = []
        if not prompt_file.exists():
            missing.append("prompt.txt")
        if not first_frame.exists():
            missing.append("first_frame.png")
        if not last_frame.exists():
            missing.append("last_frame.png")

        if missing:
            print(f"⚠️  Skipping {shot_dir.name}: missing {', '.join(missing)}")
            continue

        # Read prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        shots.append({
            'name': shot_dir.name,
            'path': shot_dir,
            'prompt': prompt,
            'first_frame': first_frame,
            'last_frame': last_frame
        })

    return shots


def prepare_shot_images(shot: Dict[str, Any], comfyui_input_dir: Path) -> Tuple[str, str]:
    """
    Copy shot images to ComfyUI input directory with unique names.
    Returns: (first_frame_filename, last_frame_filename)
    """
    shot_name = shot['name']

    # Create unique filenames
    first_filename = f"{shot_name}_first.png"
    last_filename = f"{shot_name}_last.png"

    # Copy images
    shutil.copy(shot['first_frame'], comfyui_input_dir / first_filename)
    shutil.copy(shot['last_frame'], comfyui_input_dir / last_filename)

    return first_filename, last_filename


def inject_shot_into_workflow(base_workflow: Dict, shot: Dict[str, Any],
                               first_img: str, last_img: str,
                               negative_prompt: Optional[str] = None) -> Dict:
    """
    Inject shot-specific data into workflow.
    Modifies: prompt (node 6), first_frame (node 68), last_frame (node 62)
    """
    workflow = copy.deepcopy(base_workflow)

    # Inject positive prompt
    workflow['6']['inputs']['text'] = shot['prompt']

    # Inject negative prompt if provided
    if negative_prompt:
        workflow['7']['inputs']['text'] = negative_prompt

    # Inject image filenames
    workflow['68']['inputs']['image'] = first_img
    workflow['62']['inputs']['image'] = last_img

    return workflow


# ========================================
# WORKFLOW EXECUTION (from batch_runner_v2)
# ========================================

def run_workflow_simple(workflow: Dict, server_address: str, poll_interval: int) -> Tuple[bool, str]:
    """Execute workflow and return (success, prompt_id)."""
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

    def is_complete(prompt_id):
        history = get_history(prompt_id)
        if prompt_id in history:
            return 'outputs' in history[prompt_id] or 'status' in history[prompt_id]
        return False

    try:
        prompt_id = queue_prompt(workflow)
        while not is_complete(prompt_id):
            time.sleep(poll_interval)
        return True, prompt_id
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return False, ""


def download_outputs(prompt_id: str, output_dir: Path, server_address: str) -> List[Path]:
    """Download output files from ComfyUI."""
    import urllib.request
    import urllib.parse

    def get_history(prompt_id):
        url = f"http://{server_address}/history/{prompt_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def download_file(filename, subfolder, folder_type, save_path):
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"http://{server_address}/view?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:
            with open(save_path, 'wb') as f:
                while chunk := response.read(8192):
                    f.write(chunk)

    history = get_history(prompt_id)
    outputs = history[prompt_id].get('outputs', {})
    saved_files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for node_id, node_output in outputs.items():
        for media_key in ['images', 'gifs', 'videos']:
            if media_key in node_output:
                for media in node_output[media_key]:
                    save_path = output_dir / media['filename']
                    download_file(media['filename'], media.get('subfolder', ''),
                                media.get('type', 'output'), save_path)
                    saved_files.append(save_path)

    return saved_files


# ========================================
# MAIN BATCH EXECUTION
# ========================================

def run_shot_batch(config_path: str):
    """Main shot-based batch execution function."""
    print("=" * 70)
    print("🎬 Shot-Based Batch Parameter Explorer")
    print("=" * 70)

    config = load_config(config_path)
    print(f"\n✓ Loaded config: {config_path}")

    # Load workflow
    workflow_file = Path(config['workflow_file'])
    with open(workflow_file, 'r') as f:
        base_workflow = json.load(f)
    print(f"✓ Loaded workflow: {workflow_file}")

    # Discover shots
    shots_config = config.get('shots', {})
    shots_dir = Path(shots_config.get('directory', 'shots')).expanduser()
    comfyui_input = Path(shots_config.get('comfyui_input_path', '~/comfy/ComfyUI/input')).expanduser()
    negative_prompt = shots_config.get('negative_prompt', None)

    print(f"\n📁 Scanning shots directory: {shots_dir}")
    shots = discover_shots(shots_dir)

    if not shots:
        print("❌ No valid shots found!")
        sys.exit(1)

    print(f"✓ Found {len(shots)} valid shot(s):")
    for shot in shots:
        print(f"   • {shot['name']}: \"{shot['prompt'][:60]}...\"")

    # Generate parameter samples
    param_samples, param_specs = generate_parameter_samples(config)
    seeds_per_sample = config.get('seeds_per_sample', 1)
    seed_params = [k for k, v in param_specs.items() if v['type'] == 'random_seed']

    # Setup output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_dir = Path(config['output']['base_dir']) / f"batch_shots_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(batch_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    server_address = config['comfyui']['server_address']
    poll_interval = config['comfyui']['poll_interval']

    total_per_shot = len(param_samples) * seeds_per_sample
    total_runs = len(shots) * total_per_shot

    print(f"\n🚀 Starting batch execution:")
    print(f"   Shots: {len(shots)}")
    print(f"   Parameter combinations: {len(param_samples)}")
    print(f"   Seeds per combination: {seeds_per_sample}")
    print(f"   Total runs: {total_runs}")
    print("=" * 70)

    overall_run_id = 0

    # Process each shot
    for shot_idx, shot in enumerate(shots, 1):
        print(f"\n{'='*70}")
        print(f"📷 SHOT {shot_idx}/{len(shots)}: {shot['name']}")
        print(f"{'='*70}")
        print(f"Prompt: \"{shot['prompt'][:80]}...\"")

        # Prepare shot images
        print(f"\n📋 Preparing shot images...")
        first_img, last_img = prepare_shot_images(shot, comfyui_input)
        print(f"   ✓ Copied: {first_img}, {last_img}")

        # Create shot output directory
        shot_dir = batch_dir / shot['name']
        runs_dir = shot_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Setup metadata CSV for this shot
        metadata_file = shot_dir / "metadata.csv"
        param_names = list(param_specs.keys())
        csv_file = open(metadata_file, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=['run_id'] + param_names +
                                    ['prompt_id', 'output_path', 'status', 'timestamp'])
        csv_writer.writeheader()

        # Inject shot into workflow
        shot_workflow = inject_shot_into_workflow(base_workflow, shot, first_img, last_img, negative_prompt)

        # Run parameter variations for this shot
        run_id = 0
        for sample_idx, params in enumerate(param_samples):
            for seed_idx in range(seeds_per_sample):
                run_id += 1
                overall_run_id += 1

                # Add random seeds
                run_params = dict(params)
                for seed_param in seed_params:
                    run_params[seed_param] = random.randint(0, 2**32 - 1)

                run_name = f"{run_id:04d}"
                run_dir = runs_dir / run_name

                print(f"\n[Shot {shot_idx}/{len(shots)}] [Run {run_id}/{total_per_shot}] ▶️  {run_name}")
                print(f"   Params: {run_params}")

                # Modify workflow with parameters
                workflow = modify_workflow_generic(shot_workflow, run_params, param_specs)

                # Execute
                success, prompt_id = run_workflow_simple(workflow, server_address, poll_interval)

                if success:
                    print(f"   ✓ Execution complete (prompt_id: {prompt_id})")
                    print(f"   📥 Downloading outputs...")
                    output_files = download_outputs(prompt_id, run_dir, server_address)

                    if output_files:
                        print(f"   ✓ Saved {len(output_files)} file(s)")

                        # Save params JSON
                        with open(run_dir / "params.json", 'w') as f:
                            json.dump(run_params, f, indent=2)

                        # Write metadata
                        row = {'run_id': run_id, 'prompt_id': prompt_id,
                              'output_path': str(run_dir), 'status': 'success',
                              'timestamp': datetime.now().isoformat()}
                        for pname in param_names:
                            row[pname] = run_params.get(pname, '')
                        csv_writer.writerow(row)
                        csv_file.flush()
                else:
                    print(f"   ❌ Failed")

        csv_file.close()
        print(f"\n✅ Shot '{shot['name']}' complete: {run_id} runs")

    print("\n" + "=" * 70)
    print("🎉 All shots complete!")
    print("=" * 70)
    print(f"   Total runs: {overall_run_id}")
    print(f"   Output: {batch_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
