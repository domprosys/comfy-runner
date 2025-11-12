#!/usr/bin/env python3
"""
Category Template Generator for ComfyUI Workflows

Automatically generates Category 1/2/3 batch configs based on workflow analysis.
Detects workflow type (base model, Light LoRA, SNR) and generates appropriate templates.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re


# Sampler group definitions
SAMPLER_GROUPS = {
    'euler_family': ['euler', 'euler_cfg_pp', 'euler_ancestral', 'euler_ancestral_cfg_pp'],
    'high_order': ['heun', 'heunpp2', 'uni_pc', 'uni_pc_bh2', 'ipndm', 'ipndm_v'],
    'dpmpp_family': ['dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu', 'dpmpp_3m_sde', 'uni_pc', 'uni_pc_bh2'],
}


def load_sampler_groups(sampler_groups_file: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load sampler groups from SAMPLER_GROUPS.md or return built-in defaults.

    Args:
        sampler_groups_file: Path to SAMPLER_GROUPS.md (optional)

    Returns:
        Dict mapping group names to sampler lists
    """
    if sampler_groups_file and sampler_groups_file.exists():
        groups = {}
        current_group = None

        with open(sampler_groups_file, 'r') as f:
            for line in f:
                # Match group headers like "## Group 1: Euler Family"
                group_match = re.match(r'^##\s+Group\s+\d+:\s+(.+)$', line)
                if group_match:
                    group_name = group_match.group(1).strip().lower().replace(' ', '_')
                    current_group = group_name
                    groups[current_group] = []
                    continue

                # Match sampler lines like "  - euler"
                if current_group:
                    sampler_match = re.match(r'^\s+-\s+(\w+)', line)
                    if sampler_match:
                        sampler = sampler_match.group(1).strip()
                        # Remove comments
                        if '#' not in sampler:
                            groups[current_group].append(sampler)

        return groups if groups else SAMPLER_GROUPS

    return SAMPLER_GROUPS


class WorkflowTypeDetector:
    """Detects workflow type and characteristics"""

    def __init__(self, workflow: Dict, categories: Dict[str, List[Dict]]):
        self.workflow = workflow
        self.categories = categories
        self.has_lora = self._detect_lora()
        self.has_snr = self._detect_snr()
        self.step_count = self._detect_step_count()
        self.workflow_type = self._determine_type()

    def _detect_lora(self) -> bool:
        """Detect if workflow uses LoRA"""
        for node_id, node in self.workflow.items():
            if isinstance(node, dict):
                class_type = node.get('class_type', '').lower()
                if 'lora' in class_type:
                    return True
        return False

    def _detect_snr(self) -> bool:
        """Detect if workflow uses SNR/dynamic scheduling"""
        for node_id, node in self.workflow.items():
            if isinstance(node, dict):
                class_type = node.get('class_type', '').lower()
                if any(x in class_type for x in ['snr', 'alignyoursteps', 'gits']):
                    return True
        return False

    def _detect_step_count(self) -> int:
        """Detect typical step count from workflow"""
        for param in self.categories.get('numeric', []):
            if 'steps' in param['name'].lower():
                return int(param['value'])
        return 20  # Default

    def _determine_type(self) -> str:
        """Determine workflow type"""
        if self.has_snr:
            return "snr"
        elif self.has_lora and self.step_count <= 6:
            return "light_lora"
        else:
            return "base"

    def get_recommended_samplers(self, sampler_group: Optional[str] = None) -> List[str]:
        """
        Get recommended samplers based on workflow type or specified group.

        Args:
            sampler_group: Optional group name (e.g., 'euler_family', 'dpmpp_family')

        Returns:
            List of sampler names
        """
        # If specific group requested, return it
        if sampler_group:
            groups = load_sampler_groups()
            if sampler_group in groups:
                return groups[sampler_group]
            else:
                print(f"⚠️  Unknown sampler group '{sampler_group}', using defaults")

        # Otherwise, return workflow-type defaults
        if self.workflow_type == "light_lora":
            return ["euler", "dpmpp_2m", "res_3m_ode", "res_3s_ode", "deis_2m_ode"]
        elif self.workflow_type == "base":
            # Samplers that work well with 15-30 steps
            return ["euler", "dpmpp_2m", "dpmpp_2m_sde", "heun", "dpm_2", "uni_pc", "dpmpp_3m_sde"]
        else:  # SNR
            return ["euler", "dpmpp_2m"]

    def get_recommended_schedulers(self) -> List[str]:
        """Get recommended schedulers"""
        return ["normal", "karras", "exponential", "bong_tangent"]

    @staticmethod
    def list_sampler_groups() -> Dict[str, List[str]]:
        """List all available sampler groups"""
        return load_sampler_groups()

    def get_workflow_context(self) -> str:
        """Get human-readable context about detected workflow"""
        if self.workflow_type == "light_lora":
            return f"Light LoRA workflow → Fast generation (4-6 steps), 4-10x speedup over base"
        elif self.workflow_type == "base":
            return f"Base model (no LoRA) → Higher steps ({self.step_count}), slower, maximum quality"
        else:
            return "SNR-based dynamic scheduling → Adaptive quality/speed tradeoff"


def find_sampler_nodes(categories: Dict[str, List[Dict]]) -> List[Dict]:
    """Find all sampler-related nodes"""
    sampler_nodes = []
    for param in categories.get('numeric', []):
        if 'sampler' in param['class_type'].lower() and 'steps' in param['name'].lower():
            node_id = param['path'].split('.')[1]
            sampler_nodes.append({
                'node_id': node_id,
                'class_type': param['class_type'],
                'steps_path': param['path'],
                'steps_value': param['value']
            })
    return sampler_nodes


def find_lora_nodes(categories: Dict[str, List[Dict]]) -> List[Dict]:
    """Find all LoRA strength parameters"""
    lora_params = []
    for param in categories.get('strength', []):
        if 'lora' in param['class_type'].lower():
            lora_params.append(param)
    return lora_params


def clean_numeric_value(value: Any) -> Any:
    """Clean numeric values - round floats that are very close to integers"""
    if isinstance(value, float):
        # If float is very close to an integer, convert it
        if abs(value - round(value)) < 0.0001:
            return int(round(value))
        # Otherwise round to reasonable precision
        return round(value, 2)
    return value


def get_relative_workflow_path(workflow_file: Path, output_dir: Path) -> str:
    """Calculate relative path from output_dir to workflow_file"""
    try:
        # Try to get relative path
        rel_path = Path(workflow_file).relative_to(output_dir)
        return str(rel_path)
    except ValueError:
        # If not in same tree, use ../workflow.json pattern
        return f"../{workflow_file.name}"


def generate_category2_config(
    workflow_file: Path,
    detector: WorkflowTypeDetector,
    categories: Dict[str, List[Dict]]
) -> str:
    """Generate Category 2 (Sampler Surfing) config"""

    sampler_nodes = find_sampler_nodes(categories)
    if not sampler_nodes:
        return ""

    primary_node = sampler_nodes[0]
    node_id = primary_node['node_id']

    samplers = detector.get_recommended_samplers()
    schedulers = detector.get_recommended_schedulers()

    # Calculate relative path for workflow_file
    output_dir = workflow_file.parent / "configs"
    workflow_path = get_relative_workflow_path(workflow_file, output_dir)

    config = f"""# ============================================================================
# CATEGORY 2: 🏄 SAMPLER/SCHEDULER SURFING
# Auto-generated for: {workflow_file.name}
# Workflow type: {detector.workflow_type.upper()}
# ============================================================================
#
# PURPOSE: Find the best sampler and scheduler combination
#
# DETECTED CHARACTERISTICS:
# - Type: {detector.workflow_type.upper()}
# - {detector.get_workflow_context()}
# - Step count: {detector.step_count}
# - Has LoRA: {'Yes' if detector.has_lora else 'No'}
# - Has SNR: {'Yes' if detector.has_snr else 'No'}
#
# EXPECTED OUTPUT:
#   {len(samplers)} samplers × {len(schedulers)} schedulers × 2 seeds = {len(samplers) * len(schedulers) * 2} videos
#
# USAGE:
#   cr-batch <path-to-this-file>
#
# ============================================================================

workflow_file: {workflow_path}
sampling_strategy: grid
num_samples: 100
seeds_per_sample: 2

parameters:
  # Sampler selection
  sampler_name:
    path: nodes.{node_id}.inputs.sampler_name
    type: values
    values: {samplers}

  # Scheduler selection
  scheduler:
    path: nodes.{node_id}.inputs.scheduler
    type: values
    values: {schedulers}
"""

    # Add additional sampler nodes if found (linked to primary node)
    for i, snode in enumerate(sampler_nodes[1:], start=2):
        config += f"""
  # Mirror for node {snode['node_id']} (linked to primary)
  sampler_name_node{snode['node_id']}:
    path: nodes.{snode['node_id']}.inputs.sampler_name
    type: linked
    source: sampler_name

  scheduler_node{snode['node_id']}:
    path: nodes.{snode['node_id']}.inputs.scheduler
    type: linked
    source: scheduler
"""

    # Add fixed parameters
    config += f"""
  # Fixed parameters
"""

    # Steps
    for snode in sampler_nodes:
        clean_steps = clean_numeric_value(snode['steps_value'])
        config += f"""  sampler_steps_node{snode['node_id']}:
    path: {snode['steps_path']}
    type: values
    values: [{clean_steps}]
"""

    # Add other numeric params as fixed
    for param in categories.get('numeric', []):
        if 'steps' not in param['name'].lower() and 'seed' not in param['name'].lower():
            # Skip sampler_name and scheduler
            if param['name'].lower() in ['sampler_name', 'scheduler']:
                continue
            node_id_param = param['path'].split('.')[1]
            safe_name = param['name'].replace('-', '_')
            clean_value = clean_numeric_value(param['value'])
            config += f"""  {param['class_type'].lower()}_{safe_name}_node{node_id_param}:
    path: {param['path']}
    type: values
    values: [{clean_value}]
"""

    # Add LoRA strengths if present
    lora_params = find_lora_nodes(categories)
    if lora_params:
        config += "\n  # LoRA strengths\n"
        for lora in lora_params:
            node_id_lora = lora['path'].split('.')[1]
            clean_lora = clean_numeric_value(lora['value'])
            config += f"""  lora_strength_node{node_id_lora}:
    path: {lora['path']}
    type: values
    values: [{clean_lora}]
"""

    # Add seeds
    config += "\n  # Random seeds\n"
    for param in categories.get('seeds', []):
        node_id_seed = param['path'].split('.')[1]
        config += f"""  seed_node{node_id_seed}:
    path: {param['path']}
    type: random_seed
"""

    config += """
output:
  base_dir: output
  run_name_pattern: '{run_id:04d}_s{sampler_name}_sch{scheduler}'
  save_params_json: true
  save_metadata_csv: true

resume:
  enabled: true
  skip_existing: true

comfyui:
  server_address: 127.0.0.1:8188
  poll_interval: 2
"""

    return config


def generate_category3_config(
    workflow_file: Path,
    detector: WorkflowTypeDetector,
    categories: Dict[str, List[Dict]]
) -> str:
    """Generate Category 3 (Parameter Search) config"""

    # Calculate relative path for workflow_file
    output_dir = workflow_file.parent / "configs"
    workflow_path = get_relative_workflow_path(workflow_file, output_dir)

    config = f"""# ============================================================================
# CATEGORY 3: 🔬 PARAMETER SEARCH (Sobol Strategy)
# Auto-generated for: {workflow_file.name}
# ============================================================================
#
# PURPOSE: Optimize parameters after finding best sampler/scheduler
#
# USAGE:
#   1. Fill in your winner sampler/scheduler from Category 2
#   2. cr-batch <path-to-this-file>
#
# ============================================================================

workflow_file: {workflow_path}
sampling_strategy: sobol
num_samples: 50
seeds_per_sample: 2

parameters:
  # ⚠️ EDIT HERE: Add winners from Category 2
  sampler_name:
    path: nodes.EDIT_ME.inputs.sampler_name
    type: values
    values: ["EDIT_ME"]

  scheduler:
    path: nodes.EDIT_ME.inputs.scheduler
    type: values
    values: ["EDIT_ME"]
"""

    # Add numeric parameters to vary
    config += "\n  # Parameters to optimize\n"

    # Add CFG if found
    cfg_params = [p for p in categories.get('numeric', []) if 'cfg' in p['name'].lower()]
    for cfg in cfg_params:
        node_id = cfg['path'].split('.')[1]
        config += f"""  cfg_node{node_id}:
    path: {cfg['path']}
    type: continuous
    min: 1.0
    max: 3.0
"""

    # Add shift if found
    shift_params = [p for p in categories.get('numeric', []) if 'shift' in p['name'].lower()]
    for shift in shift_params:
        node_id = shift['path'].split('.')[1]
        config += f"""  shift_node{node_id}:
    path: {shift['path']}
    type: continuous
    min: 3
    max: 7
"""

    # Add LoRA strengths
    lora_params = find_lora_nodes(categories)
    if lora_params:
        config += "\n  # LoRA strengths to optimize\n"
        for lora in lora_params:
            node_id = lora['path'].split('.')[1]
            config += f"""  lora_strength_node{node_id}:
    path: {lora['path']}
    type: continuous
    min: 0.5
    max: 3.0
"""

    # Fixed params (seeds)
    config += "\n  # Seeds\n"
    for param in categories.get('seeds', []):
        node_id = param['path'].split('.')[1]
        config += f"""  seed_node{node_id}:
    path: {param['path']}
    type: random_seed
"""

    config += """
output:
  base_dir: output
  run_name_pattern: '{run_id:04d}'
  save_params_json: true
  save_metadata_csv: true

resume:
  enabled: true
  skip_existing: true

comfyui:
  server_address: 127.0.0.1:8188
  poll_interval: 2
"""

    return config


def generate_category1_config(
    workflow_file: Path,
    detector: WorkflowTypeDetector,
    categories: Dict[str, List[Dict]]
) -> str:
    """Generate Category 1 (Seed Mining) config"""

    # Calculate relative path for workflow_file
    output_dir = workflow_file.parent / "configs"
    workflow_path = get_relative_workflow_path(workflow_file, output_dir)

    config = f"""# ============================================================================
# CATEGORY 1: 🎲 SEED MINING
# Auto-generated for: {workflow_file.name}
# ============================================================================
#
# PURPOSE: Generate variations to find hero seeds
#
# USAGE:
#   1. Fill in ALL winners from Categories 2 & 3
#   2. cr-batch <path-to-this-file>
#
# ============================================================================

workflow_file: {workflow_path}
sampling_strategy: grid
num_samples: 100
seeds_per_sample: 20

parameters:
  # ⚠️ EDIT HERE: Add winners from Category 2
  sampler_name:
    path: nodes.EDIT_ME.inputs.sampler_name
    type: values
    values: ["EDIT_ME"]

  scheduler:
    path: nodes.EDIT_ME.inputs.scheduler
    type: values
    values: ["EDIT_ME"]

  # ⚠️ EDIT HERE: Add optimal values from Category 3
"""

    # Add all numeric params as placeholders
    for param in categories.get('numeric', []):
        if 'steps' in param['name'].lower() or 'seed' in param['name'].lower():
            continue
        node_id = param['path'].split('.')[1]
        safe_name = param['name'].replace('-', '_')
        config += f"""  {safe_name}_node{node_id}:
    path: {param['path']}
    type: values
    values: [EDIT_ME]  # Replace with optimal value
"""

    # Add LoRA
    lora_params = find_lora_nodes(categories)
    if lora_params:
        for lora in lora_params:
            node_id = lora['path'].split('.')[1]
            config += f"""  lora_strength_node{node_id}:
    path: {lora['path']}
    type: values
    values: [EDIT_ME]  # Replace with optimal value
"""

    # Seeds (these vary!)
    config += "\n  # Seeds (these will vary)\n"
    for param in categories.get('seeds', []):
        node_id = param['path'].split('.')[1]
        config += f"""  seed_node{node_id}:
    path: {param['path']}
    type: random_seed
"""

    config += """
output:
  base_dir: output
  run_name_pattern: '{run_id:04d}_seed{seed_node...}'
  save_params_json: true
  save_metadata_csv: true

resume:
  enabled: true
  skip_existing: true

comfyui:
  server_address: 127.0.0.1:8188
  poll_interval: 2
"""

    return config


def generate_all_category_templates(
    workflow_file: Path,
    workflow: Dict,
    categories: Dict[str, List[Dict]],
    output_dir: Path = None
) -> Dict[str, Path]:
    """Generate all category templates and save them"""

    if output_dir is None:
        output_dir = workflow_file.parent / "configs"

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = WorkflowTypeDetector(workflow, categories)

    # Generate configs
    cat2_content = generate_category2_config(workflow_file, detector, categories)
    cat3_content = generate_category3_config(workflow_file, detector, categories)
    cat1_content = generate_category1_config(workflow_file, detector, categories)

    # Save files
    cat2_path = output_dir / "category2_sampler_surfing.yaml"
    cat3_path = output_dir / "category3_parameter_search.yaml"
    cat1_path = output_dir / "category1_seed_mining.yaml"

    with open(cat2_path, 'w') as f:
        f.write(cat2_content)

    with open(cat3_path, 'w') as f:
        f.write(cat3_content)

    with open(cat1_path, 'w') as f:
        f.write(cat1_content)

    return {
        'category1': cat1_path,
        'category2': cat2_path,
        'category3': cat3_path,
        'workflow_type': detector.workflow_type,
        'has_lora': detector.has_lora,
        'step_count': detector.step_count
    }
