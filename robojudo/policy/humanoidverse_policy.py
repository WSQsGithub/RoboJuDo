from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from robojudo.deployment.humanoidverse.deployer import HumanoidVerseDeployer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import HumanoidVersePolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class HumanoidVersePolicy(Policy):
    cfg_policy: HumanoidVersePolicyCfg

    def __init__(self, cfg_policy: HumanoidVersePolicyCfg, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.commands_map = self.cfg_policy.commands_map
        self._deployer = HumanoidVerseDeployer(self._build_hydra_config())
        self._cached_commands = np.zeros(3, dtype=np.float32)
        self.reset()

    def _build_hydra_config(self) -> DictConfig:
        config_path = Path(self.cfg_policy.template_cfg).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"HumanoidVerse template config not found: {config_path}")

        overrides = [
            f"runtime.backend={self.cfg_policy.runtime_backend}",
            f"runtime.action_dim={self.num_actions}",
            f"runtime.output_name={self.cfg_policy.onnx_output_name}",
        ]

        if self.cfg_policy.model_path is not None:
            overrides.append(f"runtime.model_path={self.cfg_policy.model_path}")

        overrides.extend(self.cfg_policy.hydra_overrides)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(version_base=None, config_dir=config_path.parent.as_posix()):
            cfg = compose(config_name=config_path.stem, overrides=overrides)

        cfg.runtime.policy_input_key = self.cfg_policy.policy_input_key
        cfg.runtime.input_name = self.cfg_policy.onnx_input_name

        if self.cfg_policy.policy_input_keys is not None:
            cfg.runtime.policy_input_keys = list(self.cfg_policy.policy_input_keys)
        if self.cfg_policy.onnx_input_names is not None:
            cfg.runtime.input_names = list(self.cfg_policy.onnx_input_names)
        if self.cfg_policy.input_shapes is not None:
            cfg.runtime.input_shapes = self.cfg_policy.input_shapes

        return OmegaConf.create(cfg)

    def _get_commands(self, ctrl_data) -> np.ndarray:
        commands = np.zeros(3, dtype=np.float32)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(lx, self.commands_map[1])
                commands[2] = command_remap(rx, self.commands_map[2])
                return commands

            if key == "KeyboardCtrl":
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] != "keyboard":
                        continue
                    value = event["pressed"] * 1.5
                    match event["name"]:
                        case "w":
                            commands[0] = command_remap(value, self.commands_map[0])
                        case "s":
                            commands[0] = command_remap(-value, self.commands_map[0])
                        case "a":
                            commands[1] = command_remap(-value, self.commands_map[1])
                        case "d":
                            commands[1] = command_remap(value, self.commands_map[1])
                        case "e":
                            commands[2] = command_remap(value, self.commands_map[2])
                        case "q":
                            commands[2] = command_remap(-value, self.commands_map[2])
                return commands

        return commands

    def _build_raw_obs(self, env_data, ctrl_data) -> dict[str, np.ndarray]:
        self._cached_commands = np.asarray(self._get_commands(ctrl_data), dtype=np.float32)
        raw_obs: dict[str, np.ndarray] = {}
        for term_name in self._deployer.obs_packer.term_names():
            getter_name = f"_get_obs_{term_name}"
            getter = getattr(self, getter_name, None)
            if getter is None:
                raise KeyError(
                    f"Missing obs getter '{getter_name}'. Add this method in HumanoidVersePolicy to support yaml term '{term_name}'."
                )
            raw_obs[term_name] = np.asarray(getter(env_data, ctrl_data), dtype=np.float32)

        return raw_obs

    def _get_obs_base_lin_vel(self, env_data, ctrl_data):
        if env_data.base_lin_vel is None:
            return np.zeros(3, dtype=np.float32)
        return env_data.base_lin_vel

    def _get_obs_base_ang_vel(self, env_data, ctrl_data):
        return env_data.base_ang_vel

    def _get_obs_projected_gravity(self, env_data, ctrl_data):
        return get_gravity_orientation(env_data.base_quat)

    def _get_obs_command(self, env_data, ctrl_data):
        return self._cached_commands

    def _get_obs_command_lin_vel(self, env_data, ctrl_data):
        return self._cached_commands[:2]

    def _get_obs_command_ang_vel(self, env_data, ctrl_data):
        return self._cached_commands[2:3]

    def _get_obs_dof_pos(self, env_data, ctrl_data):
        return env_data.dof_pos - self.default_dof_pos

    def _get_obs_dof_vel(self, env_data, ctrl_data):
        return env_data.dof_vel

    def _get_obs_last_action(self, env_data, ctrl_data):
        return self.last_action

    def _get_obs_actions(self, env_data, ctrl_data):
        return self.last_action

    def _get_obs_base_quat(self, env_data, ctrl_data):
        return env_data.base_quat

    def reset(self):
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._stashed_action = np.zeros(self.num_actions, dtype=np.float32)

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def get_observation(self, env_data, ctrl_data):
        raw_obs = self._build_raw_obs(env_data, ctrl_data)
        action, packed_obs = self._deployer.step(raw_obs)
        self._stashed_action = action.astype(np.float32, copy=False)
        self.last_action = self._stashed_action.copy()

        extras = {
            "packed_obs": packed_obs,
        }
        dummy_obs = np.zeros(1, dtype=np.float32)
        return dummy_obs, extras

    def get_action(self, obs):
        return self._stashed_action.copy()
