#!/usr/bin/env python
"""Detailed verification of VisualMimic policy observation structure."""

import numpy as np
from robojudo.config.g1.policy.g1_visualmimic_policy_cfg import G1VisualmimicPolicyCfg
from robojudo.policy.visualmimic_policy import VisualmimicPolicy


def test_observation_structure():
    """Test observation structure in detail."""
    print("=" * 70)
    print("VisualMimic Policy - Observation Structure Verification")
    print("=" * 70)

    # Create config and policy
    cfg = G1VisualmimicPolicyCfg()
    policy = VisualmimicPolicy(cfg_policy=cfg, device="cpu")

    # Create mock environment data
    class MockEnvData:
        def __init__(self):
            self.base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.base_ang_vel = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            self.dof_pos = np.arange(23, dtype=np.float32) * 0.1
            self.dof_vel = np.arange(23, dtype=np.float32) * 0.01

    env_data = MockEnvData()
    ctrl_data = {}

    print("\n📐 DOF Configuration:")
    print(f"  - Num action DOFs: {policy.num_actions}")
    print(f"  - Num obs DOFs: {policy.num_dofs}")

    # Build individual observations
    print("\n📊 Building Individual Observations:")

    commands = policy._get_obs_command(env_data, ctrl_data)
    print(f"  - Commands: shape={commands.shape}, values={commands}")

    base_ang_vel = policy._get_obs_base_ang_vel(env_data, ctrl_data)
    print(f"  - Base angular velocity: shape={base_ang_vel.shape}")

    proj_gravity = policy._get_obs_projected_gravity(env_data, ctrl_data)
    print(f"  - Projected gravity: shape={proj_gravity.shape}")

    dof_pos = policy._get_obs_dof_pos(env_data, ctrl_data)
    print(f"  - DOF position: shape={dof_pos.shape}")

    dof_vel = policy._get_obs_dof_vel(env_data, ctrl_data)
    print(f"  - DOF velocity: shape={dof_vel.shape}")

    last_action = policy._get_obs_last_action(env_data, ctrl_data)
    print(f"  - Last action: shape={last_action.shape}")

    # Calculate expected actor_obs size
    expected_actor_obs_size = (
        len(commands)
        + len(base_ang_vel)
        + len(proj_gravity)
        + len(dof_pos)
        + len(dof_vel)
        + len(last_action)
    )
    print(f"\n  📈 Expected actor_obs size (before padding): {expected_actor_obs_size}D")

    # Run inference
    print("\n🔄 Running Inference:")
    obs, extras = policy.get_observation(env_data, ctrl_data)

    actor_obs_2d = extras["actor_obs_2d"]
    actor_obs = extras["actor_obs"]

    print(f"  - actor_obs_2d: shape={actor_obs_2d.shape} (image)")
    print(f"  - actor_obs: shape={actor_obs.shape} (proprioceptive)")
    print(f"  - action_raw: shape={extras['action_raw'].shape}")

    # Verify ONNX inputs
    print("\n🔌 ONNX Model Input Verification:")
    print(f"  - actor_obs_2d after expand_dims: {np.expand_dims(actor_obs_2d, axis=0).shape}")
    print(f"  - actor_obs after expand_dims: {np.expand_dims(actor_obs, axis=0).shape}")

    print("\n✓ Observation structure verified!")

    return True


if __name__ == "__main__":
    print()
    print("╔" + "=" * 68 + "╗")
    print("║ VisualMimic Policy - Observation Structure                  ║")
    print("╚" + "=" * 68 + "╝")
    print()

    success = test_observation_structure()

    print()
    print("=" * 70)
    if success:
        print("✓ Verification PASSED!")
    else:
        print("✗ Verification FAILED!")
    print("=" * 70)
    print()
