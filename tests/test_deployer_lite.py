#!/usr/bin/env python3
"""
Lightweight test script for deployer without full robojudo dependencies.
Tests the core deployment logic without importing torch from robojudo.__init__.
"""

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need
from robojudo.deployment.humanoidverse.deployer import HumanoidVerseDeployer


def main():
    # Load config
    cfg_path = Path("robojudo/deployment/humanoidverse/config/pnp.yaml")
    print(f"Loading config from: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Print config summary
    print(f"\n📋 Config Summary:")
    print(f"  Backend: {cfg.runtime.backend}")
    print(f"  Model path: {cfg.runtime.model_path}")
    print(f"  Action dim: {cfg.runtime.action_dim}")
    print(f"  Observation clip: {cfg.obs.clip}")
    print(f"  Obs terms: {len(cfg.obs.terms)} terms")
    print(f"  Obs groups: {list(cfg.obs.groups.keys())}")

    # Create deployer
    print(f"\n🚀 Initializing deployer with backend: {cfg.runtime.backend}")
    deployer = HumanoidVerseDeployer(cfg)
    print("✓ Deployer initialized successfully!")

    # Generate dummy observations
    print(f"\n📊 Running 3 test steps with dummy observations...")
    for step in range(3):
        # Create dummy raw observations
        raw_obs = {}
        for term_name, term_cfg in cfg.obs.terms.items():
            dim = term_cfg.dim
            raw_obs[term_name] = np.random.randn(dim).astype(np.float32)

        # Run step
        action, packed = deployer.step(raw_obs)

        print(
            f"  Step {step + 1}: "
            f"input_dim={packed[cfg.runtime.policy_input_key].shape[0]}, "
            f"action_dim={action.shape[0]}"
        )

    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    exit(main())
