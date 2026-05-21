#!/usr/bin/env python3
"""
Convert HumanoidVerse exported config to deployer compatible config.

Usage:
    python scripts/convert_humanoidverse_config.py \
        --humanoidverse-config assets/models/g1/visualmimic/pnp_config.yaml \
        --output robojudo/deployment/humanoidverse/config/pnp.yaml \
        --model-path assets/models/g1/visualmimic/pnp_model.onnx \
        --robot g1
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml


def resolve_template_variable(hv_config: Dict[str, Any], var_str: str) -> int | None:
    """Resolve HumanoidVerse template variable like ${robot.dof_obs_size}."""
    if not isinstance(var_str, str) or not var_str.startswith("${"):
        return None

    # Extract path like "robot.dof_obs_size"
    path = var_str[2:-1]  # Remove ${ and }
    parts = path.split(".")

    # Navigate through config
    value = hv_config
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None

    return int(value) if value is not None else None


def extract_obs_terms(hv_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract observation terms and their properties from HumanoidVerse config."""
    obs_config = hv_config.get("obs", {})
    obs_dims_raw = obs_config.get("obs_dims", [])
    obs_scales = obs_config.get("obs_scales", {})

    # Convert obs_dims to dict (supports both list[dict] and dict)
    obs_dims: Dict[str, Any] = {}
    if isinstance(obs_dims_raw, dict):
        obs_dims = dict(obs_dims_raw)
    elif isinstance(obs_dims_raw, list):
        for item in obs_dims_raw:
            if isinstance(item, dict):
                for key, value in item.items():
                    obs_dims[key] = value

    terms = {}
    for obs_name, dim_value in obs_dims.items():
        # Handle template variables like ${robot.dof_obs_size}
        if isinstance(dim_value, str) and dim_value.startswith("${"):
            resolved = resolve_template_variable(hv_config, dim_value)
            if resolved is None:
                # Skip if can't resolve
                continue
            dim_value = resolved

        terms[obs_name] = {
            "dim": dim_value,
            "scale": obs_scales.get(obs_name, 1.0),
        }

    return terms


def extract_actor_obs_group(hv_config: Dict[str, Any]) -> List[str]:
    """Extract the actor_obs group from HumanoidVerse config."""
    obs_config = hv_config.get("obs", {})
    obs_dict = obs_config.get("obs_dict", {})
    actor_obs_group = obs_dict.get("actor_obs", [])
    return actor_obs_group


def extract_history_config(hv_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract obs_auxiliary history configuration from HumanoidVerse config."""
    obs_aux = hv_config.get("obs", {}).get("obs_auxiliary", {})
    if not isinstance(obs_aux, dict):
        return {}

    normalized: Dict[str, Dict[str, int]] = {}
    for aux_key, aux_cfg in obs_aux.items():
        if not isinstance(aux_cfg, dict):
            continue
        normalized[aux_key] = {}
        for obs_key, repeat in aux_cfg.items():
            try:
                normalized[aux_key][obs_key] = int(repeat)
            except (TypeError, ValueError):
                continue
    return normalized


def calculate_actor_obs_size(
    actor_obs_terms: List[str],
    obs_dims: Dict[str, int],
    obs_auxiliary: Dict[str, Dict[str, int]],
) -> int:
    """Calculate total actor observation size including history."""
    total_size = 0

    # Add direct observation terms
    for term_name in actor_obs_terms:
        if term_name in obs_dims:
            total_size += obs_dims[term_name]
        elif term_name in obs_auxiliary:
            # History term - add all its components
            for hist_term, repeat in obs_auxiliary[term_name].items():
                if hist_term in obs_dims:
                    total_size += obs_dims[hist_term] * repeat

    return total_size


def load_humanoidverse_config(config_path: Path) -> Dict[str, Any]:
    """Load HumanoidVerse exported config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_deployer_config(
    hv_config: Dict[str, Any],
    model_path: str,
    robot: str = "g1",
    backend: str = "onnx",
) -> Dict[str, Any]:
    """Create deployer compatible config from HumanoidVerse config."""

    # Extract observation information
    obs_terms = extract_obs_terms(hv_config)
    history_config = extract_history_config(hv_config)

    # Get all obs groups from HumanoidVerse config
    obs_dict = hv_config.get("obs", {}).get("obs_dict", {})
    if not isinstance(obs_dict, dict):
        obs_dict = {}
    available_terms = set(obs_terms.keys())
    
    # Create groups: filter each group to only include existing terms
    groups = {}
    for group_name, group_terms in obs_dict.items():
        filtered_group = [term for term in group_terms if term in available_terms]
        if filtered_group:
            groups[group_name] = filtered_group

    # Determine primary group (policy input)
    # Usually it's 'actor_obs', but fallback to first group
    primary_group = "actor_obs" if "actor_obs" in groups else list(groups.keys())[0] if groups else "actor_obs"

    # Get action dimension from env config
    env_config = hv_config.get("env", {}).get("config", {})
    normalization = env_config.get("normalization", {})
    clip_actions = normalization.get("clip_actions", 80.0)

    # Try to infer action dim from policy or robot config
    robot_config = hv_config.get("robot", {})
    dof_obs_size = robot_config.get("dof_obs_size", 12)
    action_dim = dof_obs_size  # Default: assume action dim = dof obs size

    # Create deployer config structure
    # Note: For ONNX models with multiple inputs, users may need to manually configure
    # input_names and policy_input_keys if auto-detection fails
    deployer_config = {
        "defaults": ["base"],
        "runtime": {
            "backend": backend,
            "model_path": model_path,
            "providers": ["CPUExecutionProvider"],
            "input_name": primary_group,
            "input_names": [],  # Will auto-detect from ONNX model if empty
            "output_name": "action",
            "policy_input_key": primary_group,
            "policy_input_keys": [],  # Will use policy_input_key if empty
            "input_shapes": {},
            "action_dim": action_dim,
        },
        "obs": {
            "clip": normalization.get("clip_observations", 100.0),
            "terms": obs_terms,
            "groups": groups,  # All obs groups defined
            "history": history_config,
            "outputs": {
                primary_group: [primary_group],  # Output only primary group by default
            },
        },
        "visualmimic": {
            "obs_dict": obs_dict,
            "obs_auxiliary": history_config,
            "actor_obs_config": obs_dict.get("actor_obs", []),
            "tracker_obs_config": obs_dict.get("tracker_obs", []),
            "policy_input_keys": ["actor_obs_2d", "actor_obs"]
            if "actor_obs_2d" in obs_dict
            else [primary_group],
        },
    }

    return deployer_config


def main():
    parser = argparse.ArgumentParser(
        description="Convert HumanoidVerse exported config to deployer compatible config"
    )
    parser.add_argument(
        "--humanoidverse-config",
        type=Path,
        required=True,
        help="Path to HumanoidVerse exported config YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output deployer config YAML",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        help="Robot name (default: g1)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "dummy"],
        help="Runtime backend (default: onnx)",
    )

    args = parser.parse_args()

    # Verify input file exists
    if not args.humanoidverse_config.exists():
        print(f"Error: Input config not found: {args.humanoidverse_config}")
        return 1

    # Load HumanoidVerse config
    print(f"Loading HumanoidVerse config from: {args.humanoidverse_config}")
    hv_config = load_humanoidverse_config(args.humanoidverse_config)

    # Create deployer config
    print("Converting to deployer config...")
    deployer_config = create_deployer_config(
        hv_config,
        model_path=str(args.model_path),
        robot=args.robot,
        backend=args.backend,
    )

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output config
    print(f"Writing deployer config to: {args.output}")
    with open(args.output, "w") as f:
        yaml.dump(
            deployer_config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print("✓ Conversion complete!")
    print(f"\nTo run the model:")
    print(f"  cd /workspaces/RoboJuDo")
    print(f"  python scripts/run_humanoidverse_deploy.py -cn {args.output.stem}")

    return 0


if __name__ == "__main__":
    exit(main())
