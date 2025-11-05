#!/usr/bin/env python3
"""
Thin wrapper around the packaged analyzer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from comfy_api.analyzer import main


# Parameter types we can identify
NUMERIC_TYPES = (int, float)
STRING_TYPES = (str,)
BOOLEAN_TYPES = (bool,)


def analyze_node(node_id: str, node_data: Dict[str, Any]) -> List[Tuple[str, str, Any, str]]:
    """
    Analyze a single node and extract parameters.
    Returns list of (path, param_name, value, type) tuples.
    """
    parameters = []

    if 'inputs' not in node_data:
        return parameters

    class_type = node_data.get('class_type', 'Unknown')
    inputs = node_data['inputs']

    for param_name, value in inputs.items():
        # Skip connections to other nodes (they're lists like ["4", 0])
        if isinstance(value, list):
            continue

        # Determine parameter type and create path
        path = f"nodes.{node_id}.inputs.{param_name}"

        if isinstance(value, bool):
            param_type = 'boolean'
        elif isinstance(value, int):
            param_type = 'integer'
        elif isinstance(value, float):
            param_type = 'float'
        elif isinstance(value, str):
            # Distinguish between file paths and text
            if any(ext in value.lower() for ext in ['.safetensors', '.pt', '.ckpt', '.pth', '.png', '.jpg']):
                param_type = 'file'
            else:
                param_type = 'string'
        else:
            param_type = 'unknown'

        parameters.append((path, param_name, value, param_type, class_type))

    return parameters


def categorize_parameters(parameters: List[Tuple]) -> Dict[str, List[Dict]]:
    """
    Categorize parameters by type and suggest reasonable exploration ranges.
    """
    categories = {
        'numeric': [],      # Integers and floats (steps, cfg, etc.)
        'seeds': [],        # Random seeds
        'strength': [],     # Strength/weight parameters (0-3 range typically)
        'text': [],         # Text prompts
        'files': [],        # Model/file references
        'boolean': [],      # True/False flags
        'other': []
    }

    for path, name, value, ptype, class_type in parameters:
        param_info = {
            'path': path,
            'name': name,
            'value': value,
            'type': ptype,
            'class_type': class_type
        }

        # Categorize by name patterns and type
        name_lower = name.lower()

        if 'seed' in name_lower:
            categories['seeds'].append(param_info)
        elif 'strength' in name_lower or 'weight' in name_lower:
            categories['strength'].append(param_info)
            # Suggest range
            param_info['suggested_range'] = {'min': 0.0, 'max': 3.0, 'step': 0.5}
        elif ptype in ['integer', 'float']:
            categories['numeric'].append(param_info)
            # Suggest range based on current value
            if isinstance(value, int):
                if value <= 10:
                    param_info['suggested_range'] = {
                        'min': max(1, value - 2),
                        'max': value + 5,
                        'step': 1
                    }
                elif value <= 100:
                    param_info['suggested_range'] = {
                        'min': max(1, value - 10),
                        'max': value + 20,
                        'step': 5
                    }
                else:
                    param_info['suggested_range'] = {
                        'min': int(value * 0.5),
                        'max': int(value * 1.5),
                        'step': int(value * 0.1)
                    }
            elif isinstance(value, float):
                if value <= 1.0:
                    param_info['suggested_range'] = {
                        'min': max(0.0, value - 0.5),
                        'max': value + 1.0,
                        'step': 0.1
                    }
                else:
                    param_info['suggested_range'] = {
                        'min': max(0.0, value * 0.5),
                        'max': value * 2.0,
                        'step': value * 0.25
                    }
        elif ptype == 'string' and 'text' in class_type.lower():
            categories['text'].append(param_info)
        elif ptype == 'file':
            categories['files'].append(param_info)
        elif ptype == 'boolean':
            categories['boolean'].append(param_info)
        else:
            categories['other'].append(param_info)

    return categories


def create_readable_param_name(class_type: str, param_name: str, node_id: str, used_names: set) -> str:
    """
    Create a human-readable parameter name from class_type and param_name.
    Handles duplicates by adding node_id suffix.

    Examples:
        KSamplerAdvanced + cfg → sampler_cfg
        LoraLoaderModelOnly + strength_model → lora_strength
        ModelSamplingSD3 + shift → sampling_shift
    """
    # Simplify class_type to shorter form
    class_name = class_type.lower()

    # Common simplifications
    simplifications = {
        'ksampleradvanced': 'sampler',
        'ksampler': 'sampler',
        'loraloadermodelonly': 'lora',
        'loraloader': 'lora',
        'modelsamplingsd3': 'sampling',
        'cliptextencode': 'clip',
        'checkpointloadersimple': 'checkpoint',
        'vaeloader': 'vae',
        'vaedecode': 'vae',
        'unetloader': 'unet',
        'cliploader': 'clip',
        'emptylatentimage': 'latent',
        'createvideo': 'video',
        'savevideo': 'video',
        'saveimage': 'image',
        'loadimage': 'image'
    }

    short_class = simplifications.get(class_name, class_name.replace('loader', '').replace('advanced', ''))

    # Clean up param name
    clean_param = param_name.lower().replace('_', '')

    # Combine
    base_name = f"{short_class}_{param_name.lower()}"

    # Handle duplicates
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    else:
        # Add node suffix for duplicates
        unique_name = f"{base_name}_node{node_id}"
        used_names.add(unique_name)
        return unique_name


def generate_config_template(workflow_file: Path, categories: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Generate a batch configuration template from analyzed parameters.
    """
    config = {
        'workflow_file': str(workflow_file),
        'sampling_strategy': 'sobol',
        'num_samples': 50,
        'seeds_per_sample': 2,

        'parameters': {},

        'output': {
            'base_dir': 'output',
            'run_name_pattern': '{run_id:04d}',
            'save_params_json': True,
            'save_metadata_csv': True
        },

        'resume': {
            'enabled': True,
            'skip_existing': True
        },

        'comfyui': {
            'server_address': '127.0.0.1:8188',
            'poll_interval': 2
        }
    }

    # Track used parameter names to handle duplicates
    used_names = set()
    param_suggestions = {}

    # Helper to extract node_id from path (nodes.91.inputs.x → 91)
    def get_node_id(path: str) -> str:
        parts = path.split('.')
        return parts[1] if len(parts) > 1 else ''

    # Numeric parameters
    for param in categories['numeric']:
        if 'suggested_range' in param:
            node_id = get_node_id(param['path'])
            readable_name = create_readable_param_name(
                param['class_type'],
                param['name'],
                node_id,
                used_names
            )
            param_suggestions[readable_name] = {
                '__comment': f"{param['class_type']} (node {node_id}): {param['name']} = {param['value']}",
                'path': param['path'],
                'type': 'linear',
                **param['suggested_range']
            }

    # Strength parameters
    for param in categories['strength']:
        if 'suggested_range' in param:
            node_id = get_node_id(param['path'])
            readable_name = create_readable_param_name(
                param['class_type'],
                param['name'],
                node_id,
                used_names
            )
            param_suggestions[readable_name] = {
                '__comment': f"{param['class_type']} (node {node_id}): {param['name']} = {param['value']}",
                'path': param['path'],
                'type': 'continuous',
                'min': param['suggested_range']['min'],
                'max': param['suggested_range']['max']
            }

    # Seeds
    for param in categories['seeds']:
        node_id = get_node_id(param['path'])
        readable_name = create_readable_param_name(
            param['class_type'],
            param['name'],
            node_id,
            used_names
        )
        param_suggestions[readable_name] = {
            '__comment': f"{param['class_type']} (node {node_id}): {param['name']} - will be randomized",
            'path': param['path'],
            'type': 'random_seed'
        }

    config['parameters'] = param_suggestions

    return config


def print_analysis_report(categories: Dict[str, List[Dict]]):
    """Print a human-readable analysis report."""
    print("\n" + "=" * 70)
    print("📊 WORKFLOW ANALYSIS REPORT")
    print("=" * 70)

    total_params = sum(len(params) for params in categories.values())
    print(f"\n✓ Found {total_params} total parameters\n")

    # Numeric parameters
    if categories['numeric']:
        print(f"🔢 Numeric Parameters ({len(categories['numeric'])}):")
        for param in categories['numeric']:
            suggested = param.get('suggested_range', {})
            if suggested:
                print(f"   • {param['class_type']}.{param['name']}")
                print(f"     Current: {param['value']}")
                print(f"     Suggested: {suggested['min']} → {suggested['max']} (step: {suggested['step']})")
            else:
                print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()

    # Strength/Weight parameters
    if categories['strength']:
        print(f"💪 Strength/Weight Parameters ({len(categories['strength'])}):")
        for param in categories['strength']:
            suggested = param.get('suggested_range', {})
            print(f"   • {param['class_type']}.{param['name']}")
            print(f"     Current: {param['value']}")
            print(f"     Suggested: {suggested['min']} → {suggested['max']}")
        print()

    # Seeds
    if categories['seeds']:
        print(f"🎲 Random Seeds ({len(categories['seeds'])}):")
        for param in categories['seeds']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()

    # Text prompts
    if categories['text']:
        print(f"📝 Text Prompts ({len(categories['text'])}):")
        for param in categories['text']:
            preview = param['value'][:50] + '...' if len(param['value']) > 50 else param['value']
            print(f"   • {param['class_type']}.{param['name']}")
            print(f"     \"{preview}\"")
        print()

    # Files
    if categories['files']:
        print(f"📁 File References ({len(categories['files'])}):")
        for param in categories['files']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()

    # Boolean flags
    if categories['boolean']:
        print(f"🚩 Boolean Flags ({len(categories['boolean'])}):")
        for param in categories['boolean']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()


def analyze_workflow(workflow_file: Path, output_config: Path = None):
    """Main workflow analysis function."""
    print("=" * 70)
    print("🔍 ComfyUI Workflow Analyzer")
    print("=" * 70)
    print(f"\n📄 Analyzing: {workflow_file}")

    # Load workflow
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)

    # Extract parameters from all nodes
    all_parameters = []
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict):
            params = analyze_node(node_id, node_data)
            all_parameters.extend(params)

    # Categorize parameters
    categories = categorize_parameters(all_parameters)

    # Print report
    print_analysis_report(categories)

    # Generate config template
    config = generate_config_template(workflow_file, categories)

    # Determine output path
    if output_config is None:
        output_config = workflow_file.with_name(f"{workflow_file.stem}_batch_config.yaml")

    # Save config
    print("=" * 70)
    print("📝 GENERATED CONFIG TEMPLATE")
    print("=" * 70)
    print(f"\n✓ Saving to: {output_config}\n")

    # Custom YAML representer to handle comments
    def represent_dict(dumper, data):
        # Remove __comment keys but print them as comments
        filtered = {}
        comments = {}
        for k, v in data.items():
            if isinstance(v, dict) and '__comment' in v:
                comments[k] = v['__comment']
                filtered[k] = {key: val for key, val in v.items() if key != '__comment'}
            else:
                filtered[k] = v
        return dumper.represent_dict(filtered)

    yaml.add_representer(dict, represent_dict)

    with open(output_config, 'w') as f:
        # Write header
        f.write("# ComfyUI Batch Configuration\n")
        f.write(f"# Auto-generated from: {workflow_file.name}\n")
        f.write("#\n")
        f.write("# Parameter types:\n")
        f.write("#   continuous: {min: X, max: Y} - sampled from continuous range\n")
        f.write("#   linear: {min: X, max: Y, step: Z} - discrete steps from X to Y\n")
        f.write("#   values: [A, B, C] - specific discrete values\n")
        f.write("#   random_seed: generates random seeds automatically\n")
        f.write("#\n")
        f.write("# Uncomment and configure the parameters you want to explore below.\n\n")

        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("💡 Next steps:")
    print(f"   1. Edit {output_config}")
    print(f"   2. Uncomment parameters you want to vary")
    print(f"   3. Adjust ranges and sampling strategy")
    print(f"   4. Run: python batch_runner.py {output_config}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
