from pydantic import field_validator, model_validator

from robojudo.config import ASSETS_DIR, ROOT_DIR, Config
from robojudo.tools.tool_cfgs import DoFConfig


class PolicyCfg(Config):
    policy_type: str  # name of the policy class
    robot: str  # robot name, e.g. "g1"

    @property
    def policy_file(self) -> str:
        """path to the policy file, to be overrided in subclass"""
        policy_file = ASSETS_DIR / f"models/{self.robot}/PLCAEHOLDER.pt"
        return policy_file.as_posix()

    disable_autoload: bool = False  # if True, disable auto loading of the policy file

    freq: int = 50  # control frequency (Hz)

    obs_dof: DoFConfig
    action_dof: DoFConfig

    # action post processing
    action_scale: float = 1.0
    action_clip: float | None = None  # clip action to [-action_clip, action_clip]
    action_beta: float = 1.0  # action smoothing factor

    # history settings
    history_length: int = 0  # number of history observations to use

    # TODO
    # # upper body override settings
    # wrist_override_idxs: list[int] = []  # indices of the wrist joints to override

    @property
    def history_obs_size(self) -> int:
        """size of the history observations, to be calc in subclass"""
        return 0

    @field_validator("action_scale", "action_clip")
    def check_action_scale(cls, v):
        if v is not None and v <= 0:
            raise ValueError("action_scale must be positive")
        return v

    @model_validator(mode="after")
    def check_history(self):
        if self.history_length < 0:
            raise ValueError("history_length cannot be negative")
        if self.history_obs_size < 0:
            raise ValueError("history_obs_size cannot be negative")
        return self


class UnitreePolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        ang_vel: float = 0.25
        command: list[float] = [2.0, 2.0, 0.25]

    policy_type: str = "UnitreePolicy"
    policy_name: str = "policy"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unitree/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = None
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [0.8, 0.5, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class UnitreeWoGaitPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.2
        gravity: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        command: list[float] = [1.0, 1.0, 1.0]

    policy_type: str = "UnitreeWoGaitPolicy"
    policy_name: str = "policy_wo_gait"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unitree/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = None
    action_beta: float = 1.0

    history_length: int = 5  # number of history observations to use
    history_obs_dims: dict[str, int] = {}

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [0.8, 0.5, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class SmoothPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05
        lin_vel: float = 0.5

    policy_type: str = "SmoothPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/smooth/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 10

    @property
    def history_obs_size(self) -> int:
        history_obs_size = 2 + 3 + 3 + 2 + 2 * self.obs_dof.num_dofs + self.action_dof.num_dofs
        return history_obs_size

    cycle_time: float = 0.8

    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class H2HPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 1.0
        dof_vel: float = 1.0

    # obs_type as "v-teleop-extend-vr-max-nolinvel"
    policy_type: str = "H2HStudentPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/h2h/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = 10.0
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    use_imu_torso: bool = False
    use_dof_pos_offset: bool = False

    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 25

    @property
    def history_obs_size(self) -> int:
        history_obs_size = 2 * self.obs_dof.num_dofs + 3 + 3 + self.action_dof.num_dofs
        return history_obs_size


class AMOPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05

    policy_type: str = "AMOPolicy"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/amo/amo_jit.pt"
        return policy_file.as_posix()

    @property
    def policy_adapter_file(self) -> str:
        policy_adapter_file = ASSETS_DIR / f"models/{self.robot}/amo/adapter_jit.pt"
        return policy_adapter_file.as_posix()

    @property
    def policy_adapter_norm_file(self) -> str:
        policy_adapter_norm_file = ASSETS_DIR / f"models/{self.robot}/amo/adapter_norm_stats.pt"
        return policy_adapter_norm_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    action_scale: float = 0.25

    commands_map: list[list[float]]


class BeyondMimicPolicyCfg(PolicyCfg):
    policy_type: str = "BeyondMimicPolicy"
    disable_autoload: bool = True

    policy_name: str
    max_timestep: int = -1
    start_timestep: int = 0

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/beyondmimic/{self.policy_name}.onnx"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    action_scales: list[float]

    without_state_estimator: bool
    override_robot_anchor_pos: bool = True  # if True, drop pos fdb

    use_modelmeta_config: bool = True  # if True, use the config from modelmeta
    use_motion_from_model: bool = True  # if True, use the motion data of onnx model

    @model_validator(mode="after")
    def check_modelmeta(self):
        if self.use_motion_from_model:
            if not self.use_modelmeta_config:
                raise ValueError("use_modelmeta_config must be True when use_motion_from_model")

        return self


class AsapPolicyCfg(PolicyCfg):
    policy_type: str = "AsapPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str
    relative_path: str

    motion_length_s: float
    start_upper_body_dof_pos: list[float] | None = None  # reserved for interpolation loco to mimic

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/asap/mimic/{self.policy_name}/{self.relative_path}"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        projected_gravity: float
        # command_lin_vel: float
        # command_ang_vel: float
        # command_stand: float
        # command_base_height: float
        # ref_upper_dof_pos: float
        dof_pos: float
        dof_vel: float
        history: float
        actions: float
        # phase_time: float
        ref_motion_phase: float
        # sin_phase: float
        # cos_phase: float

    action_scale: float = 0.25
    action_clip: float | None = 100.0
    obs_scales: ObsScalesCfg

    history_length: int = 4  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """
    Note: the history obs item should be aligned with code of policy
    IMPORTANT: the key order should be SORTED when concat history obs!!!
    """

    USE_HISTORY: bool


class AsapLocoPolicyCfg(PolicyCfg):
    policy_type: str = "AsapLocoPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str
    relative_path: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/asap/dec_loco/{self.policy_name}/{self.relative_path}"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        projected_gravity: float
        command_lin_vel: float
        command_ang_vel: float
        command_stand: float
        command_base_height: float
        ref_upper_dof_pos: float
        dof_pos: float
        dof_vel: float
        history: float
        actions: float
        # phase_time: float
        ref_motion_phase: float
        sin_phase: float
        cos_phase: float

    action_scale: float = 0.25
    action_clip: float | None = 100.0
    obs_scales: ObsScalesCfg

    history_length: int = 4  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """Note: the history obs item should be aligned with code of policy"""

    USE_HISTORY: bool
    GAIT_PERIOD: float
    NUM_UPPER_BODY_JOINTS: int

    # ======= Default Command CONFIGURATION =======
    command_base_height_default: float


class KungfuBotGeneralPolicyCfg(PolicyCfg):
    policy_type: str = "KungfuBotGeneralPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/kungfubot2/{self.policy_name}.onnx"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        dof_pos: float
        dof_vel: float
        actions: float
        roll_pitch: float
        # anchor_ref_pos: float
        anchor_ref_rot: float
        next_step_ref_motion: float
        history: float
        future_motion_root_height: float
        future_motion_roll_pitch: float
        future_motion_base_lin_vel: float
        future_motion_base_yaw_vel: float
        future_motion_dof_pos: float

    action_scale: float = 0.0  # not used, scale for each dof
    action_clip: float | None = 100.0
    action_scales: list[float]
    obs_scales: ObsScalesCfg

    history_length: int = 10  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """
    Note: the history obs item should be aligned with code of policy
    IMPORTANT: the key order should be SORTED when concat history obs!!!
    """

    compatibility_old_version: bool = False
    """For old version of kungfubot general policy (before 2025-11-13 bugfix #68)"""


class TwistPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05
        dof_pos: float = 1.0

    policy_type: str = "TwistPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/twist/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 1.0

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 10

    @property
    def n_mimic_obs(self) -> int:
        return self.action_dof.num_dofs + 8

    @property
    def history_obs_size(self) -> int:
        history_obs_size = self.n_mimic_obs + 3 + 2 + 3 * self.action_dof.num_dofs
        return history_obs_size

    ankle_idx: list[int]
    mimic_obs_total_degrees: int
    mimic_obs_wrist_ids: list[int]

    @property
    def mimic_obs_other_ids(self) -> list[int]:
        return [f for f in range(self.mimic_obs_total_degrees) if f not in self.mimic_obs_wrist_ids]


class HumanoidVersePolicyCfg(PolicyCfg):
    policy_type: str = "HumanoidVersePolicy"
    disable_autoload: bool = True

    runtime_backend: str = "dummy"
    model_path: str | None = None
    template_cfg: str = (ROOT_DIR / "robojudo/deployment/humanoidverse/config/base.yaml").as_posix()
    hydra_overrides: list[str] = []

    policy_input_key: str = "actor_obs"
    policy_input_keys: list[str] | None = None
    onnx_input_name: str = "actor_obs"
    onnx_input_names: list[str] | None = None
    onnx_output_name: str = "action"
    input_shapes: dict[str, list[int]] | None = None

    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]

    @property
    def policy_file(self) -> str:
        if self.model_path is None:
            return ""
        return self.model_path


class VisualmimicPolicyCfg(PolicyCfg):
    """Configuration for VisualMimic policy with tracker + generator architecture."""
    
    policy_type: str = "VisualmimicPolicy"
    disable_autoload: bool = True
    
    # Tracker model configuration (PyTorch)
    tracker_model_name: str  # e.g., "twist_general_motion_tracker"
    
    @property
    def tracker_model_file(self) -> str:
        """Path to tracker model file (PyTorch)."""
        tracker_file = ASSETS_DIR / f"models/{self.robot}/twist/{self.tracker_model_name}.pt"
        return tracker_file.as_posix()
    
    # Generator model configuration (ONNX)
    generator_model_name: str = "pnp_generator"
    
    @property
    def generator_model_file(self) -> str:
        """Path to generator model file (ONNX)."""
        generator_file = ASSETS_DIR / f"models/{self.robot}/visualmimic/{self.generator_model_name}.onnx"
        return generator_file.as_posix()
    
    @property
    def policy_file(self) -> str:
        """Compatibility method - returns tracker model path."""
        return self.tracker_model_file
    
    # Generator action configuration
    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 1.0

    # Observation structure from training obs_dict config.
    # tracker_obs_config mirrors obs_dict.tracker_obs in pnp_config.yaml.
    tracker_obs_config: list[str] = ["generator_actions", "tracker_proprio", "long_history"]
    # actor_obs_config mirrors obs_dict.actor_obs in pnp_config.yaml.  Set in subclass.
    actor_obs_config: list[str] = []

    # Observation dimensions from training obs_dims in pnp_config.yaml.
    # Maps obs key → scalar dimension.  Used to derive sizes and zero-fill unavailable obs.
    obs_dims: dict[str, int] = {}

    # Short history structure from training obs_auxiliary.short_history.
    # Maps obs key → frame count.  Mirrors obs_auxiliary.short_history in pnp_config.yaml.
    short_history_config: dict[str, int] = {}

    # Flat observation scales matching obs_scales in training pnp_config.yaml.
    # Replaces the previous ObsScalesCfg nested class.
    obs_scales: dict[str, float] = {
        "base_ang_vel": 0.25,
        "base_rp": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 0.25,
        "generator_actions": 1.0,
        "tracker_proprio": 1.0,
        "long_history": 1.0,
        "commands": 1.0,
        "projected_gravity": 1.0,
        "short_history": 1.0,
        "ee_pos_rel": 1.0,
        "ee_rot_rel": 1.0,
        "phase_one_hot": 1.0,
    }

    # long_history mirrors obs_auxiliary.long_history in training config.
    # Keys are obs keys; values are frame counts.
    # history_length is derived as max(long_history_config.values()).
    long_history_config: dict[str, int] = {}  # e.g. {"generator_actions": 10, "tracker_proprio": 10}

    # Default generator actions used to initialize history (long_history: generator_actions key).
    # Matches robot.control.generator.default_actions in training config.
    # When empty, zeros are used.
    generator_default_actions: list[float] = []

    @model_validator(mode="after")
    def _derive_from_config(self):
        # Derive n_mimic_obs from obs_dims if available
        if self.obs_dims and "generator_actions" in self.obs_dims:
            self.n_mimic_obs = self.obs_dims["generator_actions"]
        # Derive history_length from long_history_config
        if self.long_history_config:
            self.history_length = max(self.long_history_config.values())
        return self

    # Tracker observation structure
    ankle_idx: list[int] = []
    n_mimic_obs: int = 31  # Overridden by _derive_from_config if obs_dims["generator_actions"] is set

    @property
    def history_obs_size(self) -> int:
        """Size of one history frame = sum of obs_dims for each long_history_config key."""
        if self.obs_dims and self.long_history_config:
            return sum(self.obs_dims.get(k, 0) for k in self.long_history_config.keys())
        # Fallback: manual calculation
        return self.n_mimic_obs + 3 + 2 + 3 * self.action_dof.num_dofs

    @property
    def short_history_obs_dim(self) -> int:
        """Total flattened dim of short_history = sum(obs_dims[k] * frames for k in short_history_config)."""
        if not self.short_history_config or not self.obs_dims:
            return 0
        return sum(self.obs_dims.get(k, 0) * f for k, f in self.short_history_config.items())

    @property
    def actor_obs_dim(self) -> int:
        """Total flattened dim of actor_obs, derived from actor_obs_config and obs_dims."""
        if not self.actor_obs_config or not self.obs_dims:
            return 768  # Fallback for backward compatibility
        total = 0
        for key in self.actor_obs_config:
            if key == "short_history":
                total += self.short_history_obs_dim
            else:
                total += self.obs_dims.get(key, 0)
        return total

    # Tracker output configuration
    tracker_obs_wrist_ids: list[int] = []  # indices of wrist DOFs in tracker output
    tracker_obs_total_degrees: int = 0  # total degrees in tracker motion output
    
    @property
    def tracker_obs_other_ids(self) -> list[int]:
        """Non-wrist DOF indices in tracker output."""
        return [i for i in range(self.tracker_obs_total_degrees) if i not in self.tracker_obs_wrist_ids]
    
    # Generator configuration parameters
    generator_action_mean: list[float] = []
    generator_action_std: list[float] = []
    generator_clip_std_multiplier: float = 1.64
    
    # Multi-input support for generator model
    policy_input_keys: list[str] | None = None  # e.g., ["actor_obs_2d", "actor_obs"]
    onnx_input_names: list[str] | None = None  # ONNX model input names

    # Depth image preprocessing for actor_obs_2d
    depth_clip_near: float = 0.1
    depth_clip_far: float = 5.0
    actor_obs_2d_height: int = 45
    actor_obs_2d_width: int = 80
    
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
