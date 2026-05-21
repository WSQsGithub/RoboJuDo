# VisualMimic Pipeline Guide

This guide covers the G1 VisualMimic deployment path in RoboJuDo.

## What This Pipeline Does

The VisualMimic policy runs in two stages:

1. Generator model (`pnp_generator.onnx`)
   - Inputs:
     - `actor_obs_2d` with shape `[1, 1, 45, 80]`
     - `actor_obs` with shape `[1, 768]`
   - Output:
     - 31D tracker command

2. Tracker model (`twist_general_motion_tracker.pt`)
   - Input:
     - generator command plus proprioceptive history
   - Output:
     - final robot action with 23 DOFs for G1

## Recommended Entry Point

Use the dedicated config:

```bash
cd /workspaces/RoboJuDo
source .venv_new/bin/activate
python scripts/run_pipeline.py -c g1_visualmimic
```

## Headless Container Run

If the workspace has no GUI display, run with Xvfb:

```bash
cd /workspaces/RoboJuDo
source .venv_new/bin/activate
xvfb-run -a python scripts/run_pipeline.py -c g1_visualmimic
```

## Tests

The VisualMimic test files live under `tests/`:

- [tests/test_visualmimic_inference.py](../tests/test_visualmimic_inference.py)
- [tests/test_visualmimic_obs_structure.py](../tests/test_visualmimic_obs_structure.py)

Run them directly:

```bash
cd /workspaces/RoboJuDo
source .venv_new/bin/activate
python tests/test_visualmimic_inference.py
python tests/test_visualmimic_obs_structure.py
```

## Verified Status

These checks have been run successfully in the current workspace:

- Generator model loads from `assets/models/g1/visualmimic/pnp_generator.onnx`
- Tracker model loads from `assets/models/g1/twist/twist_general_motion_tracker.pt`
- VisualMimic tests pass
- `g1_visualmimic` config loads through `ConfigManager`
- The pipeline starts under `xvfb-run`

## Troubleshooting

### Missing `mujoco_viewer`

Install the submodule dependency:

```bash
cd /workspaces/RoboJuDo
source .venv_new/bin/activate
python submodule_install.py mujoco_viewer
```

### Missing `DISPLAY`

Use `xvfb-run` in headless environments.

### Controller Input

The `g1_visualmimic` config uses keyboard control by default so it can start without a physical joystick in headless environments.
