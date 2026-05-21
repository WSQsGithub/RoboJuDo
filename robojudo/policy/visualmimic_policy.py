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

try:
    import cv2  # pyright: ignore[reportMissingImports]
except ImportError:
    cv2 = None

from robojudo.policy import policy_registry
from robojudo.policy.humanoidverse_policy import HumanoidVersePolicy
from robojudo.policy.policy_cfgs import VisualmimicPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation, quatToEuler

logger = logging.getLogger(__name__)


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

        self.commands_map = cfg_policy.commands_map

        # Load generator model (ONNX)
        self._load_generator_model()

        # Load tracker model (Torch JIT)
        self._load_tracker_model()

        # Initialize generator action configuration
        self._init_generator_config()

        self._cached_commands = np.zeros(3, dtype=np.float32)
        # _cached_generator_command holds the most recent generator output;
        # initialised to default_generator_actions so actor_obs has a sensible value on step 0.
        if self.cfg_policy.generator_default_actions:
            self._cached_generator_command = np.array(
                self.cfg_policy.generator_default_actions, dtype=np.float32
            )
        else:
            self._cached_generator_command = np.zeros(self.cfg_policy.n_mimic_obs, dtype=np.float32)

        self.obs_scales = self.cfg_policy.obs_scales  # flat dict[str, float]
        self.ankle_idx = self.cfg_policy.ankle_idx

        # --- long_history buffer (for tracker_obs) ---
        # Frame template: [generator_actions (n_mimic_obs), tracker_proprio (zeros)]
        # Training initialises generator_actions frames with default_generator_actions.
        history_obs_size = self.cfg_policy.history_obs_size
        if self.cfg_policy.generator_default_actions:
            default_gen = np.array(self.cfg_policy.generator_default_actions, dtype=np.float32)
            prop_zeros = np.zeros(history_obs_size - len(default_gen), dtype=np.float32)
            self._history_template = np.concatenate([default_gen, prop_zeros])
        else:
            self._history_template = np.zeros(history_obs_size, dtype=np.float32)
        self._init_history(self._history_template)

        # --- short_history buffer (for actor_obs) ---
        self._init_short_history()

        self._depth_window_name = "VisualMimic Depth"
        self._depth_vis_warned = False
        self._depth_save_warned = False
        self._depth_vis_available = cv2 is not None

        self.reset()

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
            self.generator_input_names = self.cfg_policy.onnx_input_names
            if self.generator_input_names is None:
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

            self.generator_action_mean = gen_mean
            self.generator_action_std = gen_std

            multiplier_std = self.cfg_policy.generator_clip_std_multiplier
            self.generator_clip_action_limit_low = gen_mean - multiplier_std * gen_std
            self.generator_clip_action_limit_high = gen_mean + multiplier_std * gen_std

            logger.debug(f"Generator action limits: [{self.generator_clip_action_limit_low[0]:.3f}, "
                        f"{self.generator_clip_action_limit_high[0]:.3f}]")
        else:
            self.generator_action_mean = np.zeros(self.num_actions, dtype=np.float32)
            self.generator_action_std = np.ones(self.num_actions, dtype=np.float32)

    def _init_short_history(self):
        """Initialise short_history deque from short_history_config and obs_dims."""
        from collections import deque as _deque
        cfg = self.cfg_policy
        if cfg.short_history_config and cfg.obs_dims:
            sh_frames = max(cfg.short_history_config.values())
            # Build initial frame: zeros for all keys except generator_actions
            frame_parts = []
            for k in cfg.short_history_config.keys():
                dim = cfg.obs_dims.get(k, 0)
                if k == "generator_actions" and cfg.generator_default_actions:
                    val = np.array(cfg.generator_default_actions, dtype=np.float32)
                    val = val * cfg.obs_scales.get(k, 1.0)
                else:
                    val = np.zeros(dim, dtype=np.float32)
                frame_parts.append(val)
            sh_template = np.concatenate(frame_parts, dtype=np.float32)
            self.short_history_buf = _deque(
                [sh_template.copy() for _ in range(sh_frames)], maxlen=sh_frames
            )
        else:
            from collections import deque as _deque
            self.short_history_buf = _deque(maxlen=0)

    def reset(self):
        """Reset policy state."""
        self.timestep = 0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._stashed_action = np.zeros(self.num_actions, dtype=np.float32)
        if self.history_length > 0:
            self._init_history(self._history_template)
        self._init_short_history()

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
        if env_data.base_lin_vel is None:
            return np.zeros(3, dtype=np.float32)
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
        """Alias: training key 'actions' maps to last_action (raw, unscaled)."""
        return self.last_action

    # --- Composite obs methods for tracker_obs ---

    def _get_obs_generator_actions(self, env_data=None, ctrl_data=None) -> np.ndarray:
        """Return latest generator command (31D, raw — scale applied by dispatcher)."""
        return self._cached_generator_command.copy()

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

    def _get_obs_long_history(self, env_data=None, ctrl_data=None) -> np.ndarray:
        """Return flattened long_history (frames * frame_dim), oldest frame first."""
        return np.array(self.history_buf, dtype=np.float32).flatten()

    def _get_obs_short_history(self, env_data=None, ctrl_data=None) -> np.ndarray:
        """Return flattened short_history (frames * frame_dim), oldest frame first."""
        return np.array(self.short_history_buf, dtype=np.float32).flatten()

    def _build_actor_obs_2d(self, env_data) -> np.ndarray:
        """Build actor_obs_2d: Visual observation for CNN head.
        
        Shape: [1, H, W] (single-channel depth image)
        Priority: use env_data.camera_depth (MuJoCo), fallback to zeros.
        
        Returns:
            Visual observation array (1xHxW), normalized to [0, 1]
        """
        height = int(self.cfg_policy.actor_obs_2d_height)
        width = int(self.cfg_policy.actor_obs_2d_width)
        depth = getattr(env_data, "camera_depth", None)

        if depth is None:
            return np.zeros((1, height, width), dtype=np.float32)

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            logger.debug("Invalid camera_depth shape: %s", getattr(depth, "shape", None))
            return np.zeros((1, height, width), dtype=np.float32)

        # MuJoCo renders in OpenGL convention (row 0 = bottom of image).
        # Flip vertically so the image matches training environment orientation
        # where the hands are visible at the bottom of the frame.
        depth = np.flipud(depth)

        # Resize by nearest-neighbor sampling to match generator input shape.
        src_h, src_w = depth.shape
        if (src_h, src_w) != (height, width):
            y_idx = np.linspace(0, src_h - 1, height).astype(np.int32)
            x_idx = np.linspace(0, src_w - 1, width).astype(np.int32)
            depth = depth[y_idx][:, x_idx]

        near = float(self.cfg_policy.depth_clip_near)
        far = float(self.cfg_policy.depth_clip_far)
        if far <= near:
            far = near + 1.0

        depth = np.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        depth = np.clip(depth, near, far)
        depth = (depth - near) / (far - near)

        return np.expand_dims(depth.astype(np.float32), axis=0)

    def _build_actor_obs(self, env_data, ctrl_data) -> np.ndarray:
        """Build actor_obs using config-driven dispatch.

        Iterates over actor_obs_config keys; calls _get_obs_{key}() when available,
        otherwise zero-fills using obs_dims[key].  Applies obs_scales[key] to each part.
        Total size is determined by actor_obs_dim (derived from obs_dims + actor_obs_config).
        """
        self._cached_commands = np.asarray(self._get_commands(ctrl_data), dtype=np.float32)

        cfg = self.cfg_policy
        obs_parts: dict[str, np.ndarray] = {}

        for key in cfg.actor_obs_config:
            if key == "short_history":
                obs = self._get_obs_short_history(env_data, ctrl_data)
            else:
                get_fn = getattr(self, f"_get_obs_{key}", None)
                if get_fn is not None:
                    obs = np.asarray(get_fn(env_data, ctrl_data), dtype=np.float32)
                else:
                    # Zero-fill for observations not available in deployment
                    obs = np.zeros(cfg.obs_dims.get(key, 0), dtype=np.float32)
                obs = obs * cfg.obs_scales.get(key, 1.0)
            obs_parts[key] = obs.astype(np.float32)

        # Update short_history buffer with the current scaled frame for next step
        if cfg.short_history_config:
            frame_parts = [
                obs_parts.get(k, np.zeros(cfg.obs_dims.get(k, 0), dtype=np.float32))
                for k in cfg.short_history_config.keys()
            ]
            self.short_history_buf.append(np.concatenate(frame_parts, dtype=np.float32))

        actor_obs = np.concatenate(
            [obs_parts[k] for k in cfg.actor_obs_config], dtype=np.float32
        )

        expected = cfg.actor_obs_dim
        if len(actor_obs) != expected:
            logger.warning(
                "actor_obs size %d != expected %d; padding/clipping to match.",
                len(actor_obs), expected,
            )
            if len(actor_obs) < expected:
                actor_obs = np.concatenate(
                    [actor_obs, np.zeros(expected - len(actor_obs), dtype=np.float32)]
                )
            else:
                actor_obs = actor_obs[:expected]

        return actor_obs

    def _build_tracker_obs(self, env_data, generator_command: np.ndarray) -> np.ndarray:
        """Build tracker_obs using config-driven dispatch over tracker_obs_config.

        Calls _get_obs_{key}(env_data) for each key in tracker_obs_config;
        applies obs_scales[key].  After reading long_history, updates the
        long_history buffer with the current [generator_actions, tracker_proprio] frame.
        Obs sizes are fully derived from obs_dims — no hardcoded lengths.
        """
        # Store generator command so _get_obs_generator_actions() returns it
        self._cached_generator_command = generator_command

        cfg = self.cfg_policy
        obs_parts: dict[str, np.ndarray] = {}

        for key in cfg.tracker_obs_config:
            get_fn = getattr(self, f"_get_obs_{key}", None)
            if get_fn is None:
                raise RuntimeError(f"No _get_obs_{key}() method found for tracker_obs")
            obs = np.asarray(get_fn(env_data), dtype=np.float32)
            obs_parts[key] = obs * cfg.obs_scales.get(key, 1.0)

        # Update long_history buffer with the current scaled frame for next step.
        # long_history was already read above before the append, preserving correct ordering.
        history_frame = np.concatenate(
            [obs_parts[k] for k in cfg.long_history_config.keys() if k in obs_parts],
            dtype=np.float32,
        )
        self.history_buf.append(history_frame)

        return np.concatenate(
            [obs_parts[k] for k in cfg.tracker_obs_config], dtype=np.float32
        )

    def get_observation(self, env_data, ctrl_data: dict) -> tuple[np.ndarray, dict]:
        """Get observation for the policy.
        
        Builds actor_obs_2d and actor_obs, runs generator inference.
        
        Args:
            env_data: Environment data
            ctrl_data: Control data dictionary
            
        Returns:
            Tuple of (observation, extras dict)
        """
        # Build both observation types
        actor_obs_2d = self._build_actor_obs_2d(env_data)  # [1, 45, 80]
        actor_obs = self._build_actor_obs(env_data, ctrl_data)  # [768]

        # Prepare input tensors for ONNX model
        # actor_obs_2d: needs to be [1, 1, 45, 80] (batch=1, channels=1, height=45, width=80)
        # actor_obs: needs to be [1, 768] (batch=1, features=768)
        onnx_inputs = {
            "actor_obs_2d": np.expand_dims(actor_obs_2d, axis=0).astype(np.float32),  # [1, 1, 45, 80]
            "actor_obs": np.expand_dims(actor_obs, axis=0).astype(np.float32),  # [1, 768]
        }

        # Run generator inference: produces 31D command for tracker
        try:
            generator_output = self.generator_session.run(None, onnx_inputs)
            # Extract action (first output) - shape [1, 31]
            generator_command = generator_output[0].squeeze(axis=0).astype(np.float32)  # [31]
        except Exception as e:
            logger.error(f"Generator inference failed: {e}")
            import traceback
            traceback.print_exc()
            generator_command = np.zeros(self.cfg_policy.n_mimic_obs, dtype=np.float32)

        # Optional command clipping using generator stats (if shape matches)
        generator_command = self._post_process_generator_command(generator_command)

        # Build tracker observation from generator command + proprioception
        tracker_obs = self._build_tracker_obs(env_data, generator_command)

        # Run tracker inference: produces final robot action (23D)
        try:
            tracker_input = torch.from_numpy(tracker_obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                tracker_action = self.tracker_model(tracker_input).cpu().numpy().squeeze(0)
        except Exception as e:
            logger.error(f"Tracker inference failed: {e}")
            import traceback
            traceback.print_exc()
            tracker_action = np.zeros(self.num_actions, dtype=np.float32)

        # Post-process final tracker action
        action = self._post_process_tracker_action(tracker_action)
        self._stashed_action = action.copy()
        self.last_action = action.copy()

        # Return dummy observation
        dummy_obs = np.zeros(1, dtype=np.float32)

        extras = {
            "actor_obs": actor_obs,
            "actor_obs_2d": actor_obs_2d,
            "generator_command": generator_command,
            "tracker_obs": tracker_obs,
            "action_raw": action,
        }

        self.visualize_depth_map(actor_obs_2d)
        
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
        if hasattr(self, "_stashed_action"):
            return self._stashed_action.copy()
        return np.zeros(self.num_actions, dtype=np.float32)

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

        depth_map_2d = np.clip(depth_map[0], 0.0, 1.0)
        depth_u8 = (depth_map_2d * 255.0).astype(np.uint8)
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
