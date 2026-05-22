"""VisualMimic policy implementation with generator + tracker pipeline.

Based on HumanoidVerse framework.
- Generator (ONNX) inputs: actor_obs_2d [1, 1, 45, 80], actor_obs [1, 768]
- Generator output: 31D command
- Tracker (TorchScript) input: generator command + proprioception history
- Tracker output: final robot action
"""

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as rt
import torch
from omegaconf import DictConfig, OmegaConf

try:
    import cv2  # pyright: ignore[reportMissingImports]
except ImportError:
    cv2 = None

from robojudo.policy import policy_registry
from robojudo.policy.humanoidverse_policy import HumanoidVersePolicy
from robojudo.policy.policy_cfgs import VisualmimicPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation, quatToEuler

logger = logging.getLogger(__name__)


class HistoryHandler:
    def __init__(self, history_config, obs_dims, device, reversed=False, num_envs=1):
        self.obs_dims = obs_dims
        self.device = device
        self.num_envs = num_envs
        self.history = {}
        self.reversed = reversed
        self.add = self._append if reversed else self._add
        self.buffer_config = {}

        for _, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
                else:
                    self.buffer_config[obs_key] = obs_num

        for key in self.buffer_config.keys():
            self.history[key] = torch.zeros(
                num_envs, self.buffer_config[key], obs_dims[key], device=self.device
            )

        logger.info("History Handler Initialized")
        for key, value in self.buffer_config.items():
            logger.info("History key=%s, len=%s", key, value)

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.0

    def _add(self, key: str, value: torch.Tensor):
        if key not in self.history:
            raise KeyError(f"Key {key} not found in history")
        val = self.history[key].clone()
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value.clone()

    def _append(self, key: str, value: torch.Tensor):
        if key not in self.history:
            raise KeyError(f"Key {key} not found in history")
        val = self.history[key].clone()
        self.history[key][:, :-1] = val[:, 1:]
        self.history[key][:, -1] = value.clone()

    def query(self, key: str):
        if key not in self.history:
            raise KeyError(f"Key {key} not found in history")
        return self.history[key].clone()


@policy_registry.register
class VisualmimicPolicy(HumanoidVersePolicy):
    """VisualMimic policy using ONNX generator model.
    
    Inherits from HumanoidVersePolicy since it's trained with HumanoidVerse framework.
    Takes actor_obs_2d (visual) + actor_obs (proprioceptive) as inputs.
    """

    cfg_policy: VisualmimicPolicyCfg

    def __init__(self, cfg_policy: VisualmimicPolicyCfg, device: str = "cpu"):
        # Initialize base Policy class first
        from robojudo.policy.base_policy import Policy

        Policy.__init__(self, cfg_policy=cfg_policy, device=device)

        self.config = self._load_train_config()
        obs_cfg = self.config.get("obs", {}) if isinstance(self.config, dict) else {}

        self.obs_dict = self.config.obs.obs_dict
        self.obs_auxiliary = self.config.obs.obs_auxiliary
        self.obs_dims = self._extract_obs_dims()
        self.obs_scales = self.config.obs.obs_scales

        self.actor_obs_config = self.obs_dict.actor_obs
        self.actor_obs_2d_config = self.obs_dict.actor_obs_2d
        self.tracker_obs_config = self.obs_dict.tracker_obs
        self.short_history_config = self.obs_auxiliary.short_history
        self.long_history_config = self.obs_auxiliary.long_history
        
        self.ankle_idx = self.cfg_policy.ankle_idx
        
        self._init_generator_config()

        # Load generator model (ONNX)
        self._load_generator_model()

        # Load tracker model (Torch JIT)
        self._load_tracker_model()

        # Initialize generator action configuration
        self.action_scale = self.config.robot.control.action_scale
        self._init_generator_config()

        self._cached_commands = np.zeros(3, dtype=np.float32)
        # _cached_generator_command holds the most recent generator output;
        # initialised to default_generator_actions so actor_obs has a sensible value on step 0.
        self._cached_generator_command = np.array(self.generator_default_actions, dtype=np.float32)

        self.ankle_idx = self.cfg_policy.ankle_idx

        _ = self.obs_auxiliary.pop("history")
    

        
            device=torch.device(self.device),
            reversed=True
        )
        
        self._depth_window_name = "VisualMimic Depth"
        self._depth_vis_warned = False
        self._depth_save_warned = False
        self._depth_vis_available = cv2 is not None
        self._warned_get_action_without_obs = False

        self.reset()
        self.history_handler = HistoryHandler(
            num_envs=1,
            history_config=history_config,
            obs_dims=self.obs_dims,
    def _load_train_config(self) -> DictConfig:
            cfg_file = getattr(self.cfg_policy, "train_config_file", None)
            if not cfg_file:
                return OmegaConf.create({})
            device=torch.device(self.device),
            cfg_path = Path(cfg_file).expanduser()
            if not cfg_path.exists():
                logger.warning("Train config file not found: %s", cfg_path)
                return OmegaConf.create({})

            try:
                loaded = OmegaConf.load(cfg_path)
                if loaded is None:
                    return OmegaConf.create({})

                if not isinstance(loaded, DictConfig):
                    loaded = OmegaConf.create(loaded)

                return loaded
            except Exception as e:
                logger.error("Failed to load/resolve train config %s: %s", cfg_path, e)
                return OmegaConf.create({})

    def _extract_obs_dims(self) -> dict[str, int]:
        data = OmegaConf.to_container(
            self.config.obs.obs_dims,
            resolve=True
        )

        return {
            k: int(v)
            for kv in data
            for k, v in kv.items()
        }

    def _calc_actor_obs_dim(self) -> int:
        total = 0
        for key in self.actor_obs_config:
            if key == "short_history":
                total += sum(
                    self.obs_dims.get(obs_key, 0) * int(frames)
                    for obs_key, frames in self.short_history_config.items()
                )
            else:
                total += self.obs_dims.get(key, 0)
        return total

    def _calc_tracker_obs_dim(self) -> int: 
        total = 0
        for key in self.tracker_obs_config:
            if key == "long_history":
                total += sum(
                    self.obs_dims.get(obs_key, 0) * int(frames)
                    for obs_key, frames in self.long_history_config.items()
                )
            else:
                total += self.obs_dims.get(key, 0)
        return total
    
    def _load_tracker_model(self):
        """Load the tracker model (TorchScript)."""
        tracker_file = Path(self.cfg_policy.tracker_model_file).expanduser()
        if not tracker_file.exists():
            raise FileNotFoundError(f"Tracker model not found: {tracker_file}")

        try:
            self.tracker_model = torch.jit.load(tracker_file.as_posix(), map_location=self.device)
            self.tracker_model.eval()
            logger.info(f"Tracker model loaded from: {tracker_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tracker model: {e}")

    def _load_generator_model(self):
        """Load the generator model (ONNX Runtime)."""
        generator_file = Path(self.cfg_policy.generator_model_file).expanduser()
        if not generator_file.exists():
            raise FileNotFoundError(f"Generator model not found: {generator_file}")

        try:
            # Create ONNX Runtime session
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self.generator_session = rt.InferenceSession(
                generator_file.as_posix(), providers=providers
            )

            # Get input and output names
            self.generator_input_names = [
                input.name for input in self.generator_session.get_inputs()
            ]

            self.generator_output_names = [
                output.name for output in self.generator_session.get_outputs()
            ]

            logger.info(f"Generator model loaded from: {generator_file}")
            logger.info(f"Generator input names: {self.generator_input_names}")
            logger.info(f"Generator output names: {self.generator_output_names}")

        except Exception as e:
            raise RuntimeError(f"Failed to load generator model: {e}")

    def _init_generator_config(self):
        """Initialize generator action normalization."""
        if self.cfg_policy.generator_action_mean and self.cfg_policy.generator_action_std:
            gen_mean = np.array(self.cfg_policy.generator_action_mean, dtype=np.float32)
            gen_std = np.array(self.cfg_policy.generator_action_std, dtype=np.float32)

        self.generator_default_actions = np.array(self.config.robot.control.generator.default_actions, dtype=np.float32)

        gen_mean = np.array(self.config.robot.control.generator.action_clip_value.mean, dtype=np.float32)
        gen_std = np.array(self.config.robot.control.generator.action_clip_value.std, dtype=np.float32)
        multiplier_std = float(self.config.robot.control.generator.action_clip_value.clip_std_multiplier)
            

        self.generator_clip_action_limit_low = gen_mean - multiplier_std * gen_std
        self.generator_clip_action_limit_high = gen_mean + multiplier_std * gen_std

        logger.debug(
            f"Generator action limits: "
            f"[{self.generator_clip_action_limit_low[0]:.3f}, "
            f"{self.generator_clip_action_limit_high[0]:.3f}]"
        )

            logger.debug(f"Generator action limits: [{self.generator_clip_action_limit_low[0]:.3f}, "
                        f"{self.generator_clip_action_limit_high[0]:.3f}]")
        else:
            self.generator_action_mean = np.zeros(self.cfg_policy.n_mimic_obs, dtype=np.float32)
            self.generator_action_std = np.ones(self.cfg_policy.n_mimic_obs, dtype=np.float32)

    def reset(self):
        """Reset policy state."""
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._stashed_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Reset the history handler for "envs" 0
        reset_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        self.history_handler.reset(reset_ids)

    def post_step_callback(self, commands=None):
        """Called after each step."""
        self.timestep += 1

    def _get_commands(self, ctrl_data) -> np.ndarray:
        """Extract commands from control data."""
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

    def _get_obs_base_lin_vel(self, env_data, ctrl_data):
        """Get base linear velocity observation."""
        return env_data.base_lin_vel

    def _get_obs_base_ang_vel(self, env_data, ctrl_data):
        """Get base angular velocity observation."""
        return env_data.base_ang_vel

    def _get_obs_projected_gravity(self, env_data, ctrl_data):
        """Get projected gravity observation."""
        return get_gravity_orientation(env_data.base_quat)

    def _get_obs_command(self, env_data, ctrl_data):
        """Get command observation (4D: forward, lateral, rotation, stand)."""
        commands_3d = self._cached_commands
        stand_cmd = np.array([1.0], dtype=np.float32)  # Always standing in deployment
        return np.concatenate([commands_3d, stand_cmd])

    def _get_obs_command_lin_vel(self, env_data, ctrl_data):
        """Get command linear velocity observation."""
        return self._cached_commands[:2]

    def _get_obs_command_ang_vel(self, env_data, ctrl_data):
        """Get command angular velocity observation."""
        return self._cached_commands[2:3]

    def _get_obs_dof_pos(self, env_data, ctrl_data):
        """Get DOF position observation."""
        return env_data.dof_pos - self.default_dof_pos

    def _get_obs_dof_vel(self, env_data, ctrl_data):
        """Get DOF velocity observation."""
        return env_data.dof_vel

    def _get_obs_last_action(self, env_data, ctrl_data):
        """Get last action observation."""
        return self.last_action

    def _get_obs_base_quat(self, env_data, ctrl_data):
        """Get base quaternion observation."""
        return env_data.base_quat

    # --- Aliases matching training obs-key names ---

    def _get_obs_commands(self, env_data, ctrl_data):
        """Alias: training key 'commands' maps to _get_obs_command."""
        return self._get_obs_command(env_data, ctrl_data)

    def _get_obs_actions(self, env_data, ctrl_data=None):
        Priority: use env_data.camera_depth (MuJoCo), fallback to zeros.
        
        Returns:
            Visual observation array (1xHxW), normalized to [0, 1]
        """
        width = self.config.robot.camera.width
        height = self.config.robot.camera.height

        depth = env_data.camera_depth.copy()

        

        if depth is None:
            return np.zeros((1, height, width), dtype=np.float32)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            logger.debug("Invalid camera_depth shape: %s", getattr(depth, "shape", None))
            return np.zeros((1, height, width), dtype=np.float32)

        # Resize by nearest-neighbor sampling to match generator input shape.
        src_h, src_w = depth.shape
        if (src_h, src_w) != (height, width):
            y_idx = np.linspace(0, src_h - 1, height).astype(np.int32)
            x_idx = np.linspace(0, src_w - 1, width).astype(np.int32)
            depth = depth[y_idx][:, x_idx]

        """Alias: training key 'actions' maps to last_action (raw, unscaled)."""
        return self.last_action

    # --- Composite obs methods for tracker_obs ---

    def _get_obs_generator_actions(self, env_data=None, ctrl_data=None) -> np.ndarray:
        """Return latest generator command (31D, raw — scale applied by dispatcher)."""
        return self._cached_generator_command.copy()

    def _get_obs_ee_pos_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("ee_pos_rel", 0), dtype=np.float32)
    def _get_obs_ee_rot_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("ee_rot_rel", 0), dtype=np.float32)
    def _get_obs_phase_one_hot(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("phase_one_hot", 0), dtype=np.float32)
    def _get_obs_box_size(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_size", 0), dtype=np.float32)
    def _get_obs_box_mass(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_mass", 0), dtype=np.float32)
    def _get_obs_box_target_pos_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_target_pos_rel", 0), dtype=np.float32)
    def _get_obs_box_pos_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_pos_rel", 0), dtype=np.float32)
    def _get_obs_box_rot_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_rot_rel", 0), dtype=np.float32)
    def _get_obs_box_yaw_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_yaw_rel", 0), dtype=np.float32)
    def _get_obs_box_lin_vel_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_lin_vel_rel", 0), dtype=np.float32)
    def _get_obs_box_ang_vel_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("box_ang_vel_rel", 0), dtype=np.float32)
    def _get_obs_ee_lin_vel_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("ee_lin_vel_rel", 0), dtype=np.float32)
    def _get_obs_ee_ang_vel_rel(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("ee_ang_vel_rel", 0), dtype=np.float32)
    def _get_obs_hand_contact_force(self, env_data, ctrl_data=None): return np.zeros(self.obs_dims.get("hand_contact_force", 0), dtype=np.float32)
    def _get_obs_ego_cam(self, env_data, ctrl_data=None):
        """Build actor_obs_2d: Visual observation for CNN head.
        
        Shape: [1, H, W] (single-channel depth image)
        near = float(self.config.robot.camera.near_plane)
        far = float(self.config.robot.camera.far_plane)

        depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        depth = np.clip(depth, near, far)

        depth = (depth - near) / (far - near)

        vis_depth = depth.copy()
        self.visualize_depth_map(vis_depth)
        
        return np.expand_dims(depth.astype(np.float32), axis=0) - 0.5



    def _get_history(self, history_config, concat_method='default'):
        history_key_list = history_config.keys()
        history_tensors = []
        history_length = list(history_config.values())[0]
        if concat_method == 'default':
            for key in history_key_list:
                history_length = history_config[key]
                history_tensor = self.history_handler.query(key)[:, :history_length]
                history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [num_env, history_length*obs_dim]
                history_tensors.append(history_tensor)
            return torch.cat(history_tensors, dim=1).squeeze()
        elif concat_method == 'frame_wise':
            for frame in range(history_length):
                frame_components = []
                for key in history_key_list:
                    frame_obs = self.history_handler.query(key)[:, frame]
                    frame_obs = frame_obs.reshape(frame_obs.shape[0], -1)
                    frame_components.append(frame_obs)
                frame_tensor = torch.cat(frame_components, dim=1)  # Shape: [num_env, obs_dim]
                history_tensors.append(frame_tensor)
            return torch.cat(history_tensors, dim=1).squeeze()  # Shape: [, history_length*obs_dim]
        else:
            raise ValueError(f"Unknown history_concat_method: {concat_method}")
    
    def _get_obs_long_history(self, env_data=None, ctrl_data=None):
        assert "long_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['long_history']
        return self._get_history(history_config)
    
    def _get_obs_short_history(self, env_data=None, ctrl_data=None):
        assert "short_history" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['short_history']
        return self._get_history(history_config)

    def _get_obs_tracker_proprio(self, env_data, ctrl_data=None) -> np.ndarray:
        """Build tracker_proprio (74D) with sub-component scaling applied internally.

        Mirrors training's _get_obs_tracker_proprio:
          base_ang_vel * obs_scales[base_ang_vel]
          base_rp      * obs_scales[base_rp]
          dof_pos      * obs_scales[dof_pos]
          dof_vel      * obs_scales[dof_vel]  (ankle zeros zeroed first)
          actions      * obs_scales[actions]

        Returns pre-scaled 74D vector.  Top-level obs_scales[tracker_proprio]=1.0 is a no-op.
        """
        scales = self.obs_scales
        ang_vel = env_data.base_ang_vel * scales.get("base_ang_vel", 0.25)
        rpy = quatToEuler(env_data.base_quat)
        base_rp = rpy[:2] * scales.get("base_rp", 1.0)
        dof_pos = (env_data.dof_pos - self.default_dof_pos) * scales.get("dof_pos", 1.0)
        dof_vel = env_data.dof_vel.copy()
        if self.ankle_idx:
            dof_vel[self.ankle_idx] = 0.0
        dof_vel *= scales.get("dof_vel", 0.05)
        actions = self.last_action * scales.get("actions", 0.25)
        return np.concatenate([ang_vel, base_rp, dof_pos, dof_vel, actions], dtype=np.float32)

    def get_observation(self, env_data, ctrl_data: dict) -> tuple[np.ndarray, dict]:
        """Get observation for the policy.
        
        Builds all obs keys, updates history handler, and constructs actor_obs_2d, actor_obs, tracker_obs.
        """
        self._cached_commands = np.asarray(self._get_commands(ctrl_data), dtype=np.float32)

        # 1. Compute all primitive observations
        # We need to collect all unique base observation keys from all groups in obs_dict
        # as well as dependencies for history.
        cfg = self.cfg_policy

        obs_groups = {
            "actor_obs": list(self.actor_obs_config),
            "actor_obs_2d": ["ego_cam"],
            "tracker_obs": list(self.tracker_obs_config),
        }
        
        obs_buf_dict = {}
        for group_keys in obs_groups.values():
            for key in group_keys:
                if key not in ["shory_history", "long_history", "history", "short_history"]:
                    get_fn = getattr(self, f"_get_obs_{key}", None)
                    if get_fn is not None:
                        obs = np.asarray(get_fn(env_data, ctrl_data), dtype=np.float32)
                    else:
                        obs = np.zeros(self.obs_dims.get(key, 0), dtype=np.float32)
                    # Apply scale
                    obs_buf_dict[key] = obs * self.obs_scales.get(key, 1.0)

        # For tracker_proprio, it is built from parts:
        if "tracker_proprio" not in obs_buf_dict:
            obs_buf_dict["tracker_proprio"] = self._get_obs_tracker_proprio(env_data, ctrl_data)
        
        tracker_obs_dim = self._calc_tracker_obs_dim()
        assert len(tracker_obs) == tracker_obs_dim
        
        # 2. Push primitive obs to HistoryHandler
        for key in self.history_handler.history.keys():
            if key in obs_buf_dict:
                val = torch.from_numpy(obs_buf_dict[key]).unsqueeze(0).to(self.device).float()
                self.history_handler.add(key, val)
        
        # 3. Compute history observations
        for hist_key in ["short_history", "long_history", "history"]:
            # Check if it is required by obs_dict groups
            is_needed = any(hist_key in group for group in obs_groups.values())
            if is_needed:
                get_fn = getattr(self, f"_get_obs_{hist_key}", None)
                if get_fn is not None:
                    obs_buf_dict[hist_key] = np.asarray(get_fn(env_data, ctrl_data), dtype=np.float32) * self.obs_scales.get(hist_key, 1.0)
        
        # 4. Construct groups
        actor_obs_2d = self._build_actor_obs_2d(env_data)  # We will manually fetch ego_cam if it's there
        if "ego_cam" in obs_groups.get("actor_obs_2d", []):
            actor_obs_2d = obs_buf_dict["ego_cam"].reshape(1, int(cfg.actor_obs_2d_height), int(cfg.actor_obs_2d_width))
            actor_obs_2d = np.expand_dims(actor_obs_2d, axis=0) # [1, 1, H, W]
        else:
            actor_obs_2d = self._build_actor_obs_2d(env_data)

        actor_obs = np.concatenate([obs_buf_dict[k] for k in obs_groups.get("actor_obs", self.actor_obs_config)], dtype=np.float32)

        # Optional length matching
        actor_obs_dim = self._calc_actor_obs_dim()
        assert len(actor_obs) == actor_obs_dim

        
        actor_obs_2d = obs_buf_dict["ego_cam"]

        onnx_inputs = {
            "actor_obs_2d": np.expand_dims(actor_obs_2d, axis=0).astype(np.float32),
            "actor_obs": np.expand_dims(actor_obs, axis=0).astype(np.float32),
        }

        # Run generator
        try:
            generator_output = self.generator_session.run(None, onnx_inputs)
            generator_command = generator_output[0].squeeze(axis=0).astype(np.float32)
        except Exception as e:
            logger.error(f"Generator inference failed: {e}")
            generator_command = np.zeros(self.cfg_policy.n_mimic_obs, dtype=np.float32)

        generator_command = self._post_process_generator_command(generator_command)
        self._cached_generator_command = generator_command

        # Now construct tracker_obs
        tracker_obs = np.concatenate([obs_buf_dict[k] for k in obs_groups.get("tracker_obs", self.tracker_obs_config)], dtype=np.float32)

        # Run tracker
        try:
            tracker_input = torch.from_numpy(tracker_obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                tracker_action = self.tracker_model(tracker_input).cpu().numpy().squeeze(0)
        except Exception as e:
            logger.error(f"Tracker inference failed: {e}")
            tracker_action = np.zeros(self.num_actions, dtype=np.float32)

        action = self._post_process_tracker_action(tracker_action)
        self._stashed_action = action.copy()
        self.last_action = action.copy()

        dummy_obs = np.zeros(1, dtype=np.float32)
        extras = {
            "actor_obs": actor_obs,
            "actor_obs_2d": actor_obs_2d,
            "generator_command": generator_command,
            "tracker_obs": tracker_obs,
            "action_raw": action,
        }

        vis_depth = actor_obs_2d
        if vis_depth.ndim == 4 and vis_depth.shape[0] == 1:
            vis_depth = vis_depth[0]
        self.visualize_depth_map(vis_depth)
        return dummy_obs, extras


    def _post_process_generator_command(self, command: np.ndarray) -> np.ndarray:
        """Post-process generator command.
        
        Args:
            command: Raw command from generator model (31D)
            
        Returns:
            Post-processed command
        """
        # Clip command based on learned statistics only when dimensions match.
        if (hasattr(self, "generator_clip_action_limit_low") and 
            hasattr(self, "generator_clip_action_limit_high")):
            if len(command) == len(self.generator_clip_action_limit_low):
                command = np.clip(
                    command,
                    self.generator_clip_action_limit_low,
                    self.generator_clip_action_limit_high,
                )
            else:
                logger.debug(
                    "Skipping generator stat clipping due to dim mismatch: "
                    f"command={len(command)}, stats={len(self.generator_clip_action_limit_low)}"
                )

        return command.astype(np.float32)

    def _post_process_tracker_action(self, action: np.ndarray) -> np.ndarray:
        """Post-process final action from tracker output."""
        if len(action) != self.num_actions:
            if len(action) > self.num_actions:
                action = action[: self.num_actions]
            else:
                action = np.pad(action, (0, self.num_actions - len(action)), mode="constant")

        # Apply smoothing factor
        action = (1 - self.action_beta) * self.last_action + self.action_beta * action

        # Apply global action clipping
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)

        # Apply action scale
        action = action * self.action_scale

        return action.astype(np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from observation.
        
        For VisualMimic, the action is already computed in get_observation,
        so this just returns the cached action.
        """
        if not self._warned_get_action_without_obs:
            logger.warning(
                "get_action called before fresh get_observation; returning last_action."
            )
            self._warned_get_action_without_obs = True

        return self.action_scale * self._stashed_action.copy()
        


    def visualize_depth_map(self, depth_map: np.ndarray):
        """Visualize the depth map with OpenCV and refresh every step.

        Args:
            depth_map: Depth map array to visualize (1xHxW), value range [0, 1].
        """
        if depth_map.ndim != 3 or depth_map.shape[0] != 1:
            logger.error("Invalid depth map shape for visualization: %s", depth_map.shape)
            return

        if not self._depth_vis_available:
            if not self._depth_vis_warned:
                logger.warning("OpenCV (cv2) is not installed, skip depth visualization.")
                self._depth_vis_warned = True
            return

        depth_map_2d = depth_map
        depth_u8 = (depth_map_2d * 255.0).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if has_display:
            cv2.imshow(self._depth_window_name, depth_color)
            cv2.waitKey(1)
            return

        # Headless fallback: dump a frame every 30 steps for offline inspection.
        if self.timestep % 30 == 0:
            output_dir = Path("logs/depth_debug")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"depth_{self.timestep:06d}.png"
            cv2.imwrite(output_file.as_posix(), depth_color)
            if not self._depth_save_warned:
                logger.warning(
                    "No display detected; saving depth frames to %s",
                    output_dir.as_posix(),
                )
                self._depth_save_warned = True

if __name__ == "__main__":
    # Simple test
    print("VisualmimicPolicy module loaded successfully")
