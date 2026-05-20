import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from robojudo.deployment.humanoidverse.deployer import HumanoidVerseDeployer

logger = logging.getLogger("robojudo")


def _make_dummy_raw_obs(cfg: DictConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    raw_obs: dict[str, np.ndarray] = {}
    for term_name, term_cfg in cfg.obs.terms.items():
        raw_obs[term_name] = rng.standard_normal(int(term_cfg.dim)).astype(np.float32)
    return raw_obs


@hydra.main(version_base=None, config_path="../robojudo/deployment/humanoidverse/config", config_name="base")
def main(cfg: DictConfig) -> None:
    deployer = HumanoidVerseDeployer(cfg)
    rng = np.random.default_rng(int(cfg.eval.seed))

    for step_idx in range(int(cfg.eval.steps)):
        raw_obs = _make_dummy_raw_obs(cfg, rng)
        action, packed = deployer.step(raw_obs)
        logger.info(
            "step=%s obs_dim=%s action_dim=%s",
            step_idx,
            packed[cfg.runtime.policy_input_key].shape[0],
            action.shape[0],
        )


if __name__ == "__main__":
    main()
