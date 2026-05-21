#!/usr/bin/env python
"""Test VisualmimicPolicy with actual ONNX model."""

import numpy as np
from robojudo.config.g1.policy.g1_visualmimic_policy_cfg import G1VisualmimicPolicyCfg
from robojudo.policy.visualmimic_policy import VisualmimicPolicy


def test_visualmimic_full_inference():
    """Test VisualmimicPolicy with full inference pipeline."""
    print("=" * 70)
    print("VisualmimicPolicy Full Inference Test")
    print("=" * 70)

    # Create config and policy
    cfg = G1VisualmimicPolicyCfg()
    print(f"\n✓ Configuration created")
    print(f"  - Robot: {cfg.robot}")
    print(f"  - Action DOF: {cfg.action_dof.num_dofs}")
    print(f"  - Obs DOF: {cfg.obs_dof.num_dofs}")

    policy = VisualmimicPolicy(cfg_policy=cfg, device="cpu")
    print(f"\n✓ Policy instantiated")
    print(f"  - Num actions: {policy.num_actions}")
    print(f"  - Num dofs: {policy.num_dofs}")

    # Create mock environment data
    class MockEnvData:
        def __init__(self):
            self.base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            self.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.dof_pos = np.zeros(23, dtype=np.float32)
            self.dof_vel = np.zeros(23, dtype=np.float32)

    env_data = MockEnvData()
    ctrl_data = {}

    # Run inference
    print(f"\n{'=' * 70}")
    print("Running Inference...")
    print(f"{'=' * 70}")

    try:
        obs, extras = policy.get_observation(env_data, ctrl_data)
        print(f"\n✓ Inference successful!")

        # Print observation details
        print(f"\n📊 Observation Details:")
        print(f"  - Dummy obs shape: {obs.shape}")
        print(f"  - actor_obs_2d shape: {extras['actor_obs_2d'].shape}")
        print(f"  - actor_obs shape: {extras['actor_obs'].shape}")
        print(f"  - action_raw shape: {extras['action_raw'].shape}")

        # Print action details
        action = policy.get_action(obs)
        print(f"\n🤖 Action Output:")
        print(f"  - Action shape: {action.shape}")
        print(f"  - Action min: {action.min():.4f}")
        print(f"  - Action max: {action.max():.4f}")
        print(f"  - Action mean: {action.mean():.4f}")
        print(f"  - Action std: {action.std():.4f}")

        # Test multiple inference steps
        print(f"\n{'=' * 70}")
        print("Running 5 Inference Steps...")
        print(f"{'=' * 70}")

        for step in range(5):
            obs, extras = policy.get_observation(env_data, ctrl_data)
            action = policy.get_action(obs)
            policy.post_step_callback()

            print(f"  Step {step + 1}: action shape={action.shape}, "
                  f"min={action.min():.4f}, max={action.max():.4f}")

        print(f"\n✓ Multiple inference steps completed successfully!")

        return True

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    print("╔" + "=" * 68 + "╗")
    print("║ VisualMimic Policy - Full Inference Test                      ║")
    print("╚" + "=" * 68 + "╝")
    print()

    success = test_visualmimic_full_inference()

    print()
    print("=" * 70)
    if success:
        print("✓ All tests PASSED!")
    else:
        print("✗ Tests FAILED!")
    print("=" * 70)
    print()
