#!/usr/bin/env python3
"""
ComfyUI Workflow Analyzer (packaged)
Analyzes workflow JSON files and extracts parameters.
Generates template batch configuration for parameter exploration.
Can also generate Category 1/2/3 templates automatically.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

try:
    from .category_generator import generate_all_category_templates
    CATEGORY_GENERATOR_AVAILABLE = True
except ImportError:
    CATEGORY_GENERATOR_AVAILABLE = False


def analyze_node(node_id: str, node_data: Dict[str, Any]) -> List[Tuple[str, str, Any, str]]:
    parameters = []
    if 'inputs' not in node_data:
        return parameters
    class_type = node_data.get('class_type', 'Unknown')
    inputs = node_data['inputs']
    for param_name, value in inputs.items():
        if isinstance(value, list):
            continue
        path = f"nodes.{node_id}.inputs.{param_name}"
        if isinstance(value, bool):
            ptype = 'boolean'
        elif isinstance(value, int):
            ptype = 'integer'
        elif isinstance(value, float):
            ptype = 'float'
        elif isinstance(value, str):
            if any(ext in value.lower() for ext in ['.safetensors', '.pt', '.ckpt', '.pth', '.png', '.jpg']):
                ptype = 'file'
            else:
                ptype = 'string'
        else:
            ptype = 'unknown'
        parameters.append((path, param_name, value, ptype, class_type))
    return parameters


def categorize_parameters(parameters: List[Tuple]) -> Dict[str, List[Dict]]:
    categories = {'numeric': [], 'seeds': [], 'strength': [], 'text': [], 'files': [], 'boolean': [], 'other': []}
    for path, name, value, ptype, class_type in parameters:
        param_info = {'path': path, 'name': name, 'value': value, 'type': ptype, 'class_type': class_type}
        name_lower = name.lower()
        if 'seed' in name_lower:
            categories['seeds'].append(param_info)
        elif 'strength' in name_lower or 'weight' in name_lower:
            categories['strength'].append(param_info)
            param_info['suggested_range'] = {'min': 0.0, 'max': 3.0, 'step': 0.5}
        elif ptype in ['integer', 'float']:
            categories['numeric'].append(param_info)
            if isinstance(value, int):
                if value <= 10:
                    param_info['suggested_range'] = {'min': max(1, value - 2), 'max': value + 5, 'step': 1}
                elif value <= 100:
                    param_info['suggested_range'] = {'min': max(1, value - 10), 'max': value + 20, 'step': 5}
                else:
                    param_info['suggested_range'] = {'min': int(value * 0.5), 'max': int(value * 1.5), 'step': int(value * 0.1)}
            elif isinstance(value, float):
                if value <= 1.0:
                    param_info['suggested_range'] = {'min': max(0.0, value - 0.5), 'max': value + 1.0, 'step': 0.1}
                else:
                    param_info['suggested_range'] = {'min': max(0.0, value * 0.5), 'max': value * 2.0, 'step': value * 0.25}
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
    class_name = class_type.lower()
    simplifications = {
        'ksampleradvanced': 'sampler', 'ksampler': 'sampler', 'loraloadermodelonly': 'lora',
        'loraloader': 'lora', 'modelsamplingsd3': 'sampling', 'cliptextencode': 'clip',
        'checkpointloadersimple': 'checkpoint', 'vaeloader': 'vae', 'vaedecode': 'vae',
        'unetloader': 'unet', 'cliploader': 'clip', 'emptylatentimage': 'latent',
        'createvideo': 'video', 'savevideo': 'video', 'saveimage': 'image', 'loadimage': 'image'
    }
    short_class = simplifications.get(class_name, class_name.replace('loader', '').replace('advanced', ''))
    base_name = f"{short_class}_{param_name.lower()}"
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    unique_name = f"{base_name}_node{node_id}"
    used_names.add(unique_name)
    return unique_name


def generate_config_template(workflow_file: Path, categories: Dict[str, List[Dict]]) -> Dict[str, Any]:
    config = {
        'workflow_file': str(workflow_file),
        'sampling_strategy': 'sobol',
        'num_samples': 50,
        'seeds_per_sample': 2,
        'parameters': {},
        'output': {'base_dir': 'output', 'run_name_pattern': '{run_id:04d}', 'save_params_json': True, 'save_metadata_csv': True},
        'resume': {'enabled': True, 'skip_existing': True},
        'comfyui': {'server_address': '127.0.0.1:8188', 'poll_interval': 2}
    }

    used_names = set()
    param_suggestions = {}

    def get_node_id(path: str) -> str:
        parts = path.split('.')
        return parts[1] if len(parts) > 1 else ''

    for param in categories['numeric']:
        if 'suggested_range' in param:
            node_id = get_node_id(param['path'])
            readable = create_readable_param_name(param['class_type'], param['name'], node_id, used_names)
            param_suggestions[readable] = {
                '__comment': f"{param['class_type']} (node {node_id}): {param['name']} = {param['value']}",
                'path': param['path'], 'type': 'linear', **param['suggested_range']
            }

    for param in categories['strength']:
        if 'suggested_range' in param:
            node_id = get_node_id(param['path'])
            readable = create_readable_param_name(param['class_type'], param['name'], node_id, used_names)
            param_suggestions[readable] = {
                '__comment': f"{param['class_type']} (node {node_id}): {param['name']} = {param['value']}",
                'path': param['path'], 'type': 'continuous', 'min': param['suggested_range']['min'], 'max': param['suggested_range']['max']
            }

    for param in categories['seeds']:
        node_id = get_node_id(param['path'])
        readable = create_readable_param_name(param['class_type'], param['name'], node_id, used_names)
        param_suggestions[readable] = {
            '__comment': f"{param['class_type']} (node {node_id}): {param['name']} - will be randomized",
            'path': param['path'], 'type': 'random_seed'
        }

    config['parameters'] = param_suggestions
    return config


def print_analysis_report(categories: Dict[str, List[Dict]]):
    print("\n" + "=" * 70)
    print("📊 WORKFLOW ANALYSIS REPORT")
    print("=" * 70)
    total_params = sum(len(params) for params in categories.values())
    print(f"\n✓ Found {total_params} total parameters\n")

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

    if categories['strength']:
        print(f"💪 Strength/Weight Parameters ({len(categories['strength'])}):")
        for param in categories['strength']:
            suggested = param.get('suggested_range', {})
            print(f"   • {param['class_type']}.{param['name']}")
            print(f"     Current: {param['value']}")
            print(f"     Suggested: {suggested['min']} → {suggested['max']}")
        print()

    if categories['seeds']:
        print(f"🎲 Random Seeds ({len(categories['seeds'])}):")
        for param in categories['seeds']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()

    if categories['text']:
        print(f"📝 Text Prompts ({len(categories['text'])}):")
        for param in categories['text']:
            preview = param['value'][:50] + '...' if len(param['value']) > 50 else param['value']
            print(f"   • {param['class_type']}.{param['name']}")
            print(f"     \"{preview}\"")
        print()

    if categories['files']:
        print(f"📁 File References ({len(categories['files'])}):")
        for param in categories['files']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()

    if categories['boolean']:
        print(f"🚩 Boolean Flags ({len(categories['boolean'])}):")
        for param in categories['boolean']:
            print(f"   • {param['class_type']}.{param['name']} = {param['value']}")
        print()


def analyze_workflow(workflow_file: Path, output_config: Path = None, generate_categories: bool = False):
    print("=" * 70)
    print("🔍 ComfyUI Workflow Analyzer")
    print("=" * 70)
    print(f"\n📄 Analyzing: {workflow_file}")
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)
    all_parameters = []
    for node_id, node_data in workflow.items():
        if isinstance(node_data, dict):
            params = analyze_node(node_id, node_data)
            all_parameters.extend(params)
    categories = categorize_parameters(all_parameters)
    print_analysis_report(categories)

    # Generate category templates if requested
    if generate_categories:
        if not CATEGORY_GENERATOR_AVAILABLE:
            print("\n⚠️  Category generator not available (import failed)")
            print("    Falling back to standard config generation only")
        else:
            print("\n" + "=" * 70)
            print("🎯 GENERATING CATEGORY TEMPLATES")
            print("=" * 70)
            result = generate_all_category_templates(workflow_file, workflow, categories)
            print(f"\n✓ Workflow type detected: {result['workflow_type'].upper()}")
            print(f"  - Has LoRA: {'Yes' if result['has_lora'] else 'No'}")
            print(f"  - Step count: {result['step_count']}")
            print(f"\n✓ Generated templates:")
            print(f"  - Category 2 (Surfing): {result['category2']}")
            print(f"  - Category 3 (Search):  {result['category3']}")
            print(f"  - Category 1 (Mining):  {result['category1']}")
            print(f"\n💡 Next steps:")
            print(f"   1. Review generated templates in configs/")
            print(f"   2. Start with: cr-batch {result['category2']}")
            print("\n" + "=" * 70)
            return  # Skip standard config generation

    # Standard config generation
    config = generate_config_template(workflow_file, categories)
    if output_config is None:
        output_config = workflow_file.with_name(f"{workflow_file.stem}_batch_config.yaml")

    print("=" * 70)
    print("📝 GENERATED CONFIG TEMPLATE")
    print("=" * 70)
    print(f"\n✓ Saving to: {output_config}\n")

    def represent_dict(dumper, data):
        filtered = {}
        for k, v in data.items():
            if isinstance(v, dict) and '__comment' in v:
                filtered[k] = {key: val for key, val in v.items() if key != '__comment'}
            else:
                filtered[k] = v
        return dumper.represent_dict(filtered)

    yaml.add_representer(dict, represent_dict)

    with open(output_config, 'w') as f:
        f.write("# ComfyUI Batch Configuration\n")
        f.write(f"# Auto-generated from: {workflow_file.name}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("💡 Next steps:")
    print(f"   1. Edit {output_config}")
    print(f"   2. Uncomment parameters to vary")
    print(f"   3. Run: python -m comfy_api.batch {output_config}")
    print(f"\n💡 Tip: Use --generate-category-templates to auto-generate Category 1/2/3 configs!")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ComfyUI workflow and generate batch configuration templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis (generates single batch config)
  cr-analyze workflow.json

  # Generate Category 1/2/3 templates automatically
  cr-analyze workflow.json --generate-category-templates

  # Specify custom output path
  cr-analyze workflow.json --output custom_config.yaml
        """
    )

    parser.add_argument('workflow', type=str, help='Path to ComfyUI workflow JSON file')
    parser.add_argument('-o', '--output', type=str, help='Output config file path (default: <workflow>_batch_config.yaml)')
    parser.add_argument('-g', '--generate-category-templates', action='store_true',
                        help='Generate Category 1/2/3 templates (Seed Mining, Sampler Surfing, Parameter Search)')

    args = parser.parse_args()

    workflow_path = Path(args.workflow)
    output_path = Path(args.output) if args.output else None

    if not workflow_path.exists():
        print(f"❌ Error: Workflow file not found: {workflow_path}")
        sys.exit(1)

    analyze_workflow(workflow_path, output_path, generate_categories=args.generate_category_templates)


if __name__ == "__main__":
    main()

