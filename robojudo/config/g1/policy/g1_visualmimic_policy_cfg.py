"""G1 VisualMimic Policy Configuration.

This configuration sets up the VisualMimic policy for G1 robot with:
- Tracker model: twist_general_motion_tracker.pt (PyTorch)
- Generator model: pnp_generator.onnx (ONNX Runtime)
"""

from robojudo.policy.policy_cfgs import VisualmimicPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1VisualmimicDoF(DoFConfig):
    """G1 robot DOF configuration (same as G1TwistDoF)."""

    joint_names: list[str] = [
        *[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
        ],
        *[
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
        ],
        *[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ],
    ]

    default_pos: list[float] | None = [
        *[-0.2, 0.0, 0.0, 0.4, -0.2, 0.0],
        *[-0.2, 0.0, 0.0, 0.4, -0.2, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.0, 0.2, 0.0, 1.2],
        *[0.0, -0.2, 0.0, 1.2],
    ]

    stiffness: list[float] | None = [
        *[100, 100, 100, 150, 40, 40],
        *[100, 100, 100, 150, 40, 40],
        *[150, 150, 150],
        *[40, 40, 40, 40],
        *[40, 40, 40, 40],
    ]

    damping: list[float] | None = [
        *[2, 2, 2, 4, 2, 2],
        *[2, 2, 2, 4, 2, 2],
        *[4, 4, 4],
        *[5, 5, 5, 5],
        *[5, 5, 5, 5],
    ]

    torque_limits: list[float] | None = [
        *[88, 139, 88, 139, 50, 50],
        *[88, 139, 88, 139, 50, 50],
        *[88, 50, 50],
        *[25, 25, 25, 25],
        *[25, 25, 25, 25],
    ]


class G1VisualmimicPolicyCfg(VisualmimicPolicyCfg):
    """G1 VisualMimic Policy Configuration."""

    robot: str = "g1"

    # Tracker configuration (uses twist_general_motion_tracker)
    tracker_model_name: str = "twist_general_motion_tracker"

    # Generator configuration (uses pnp_generator)
    generator_model_name: str = "pnp_generator"

    # DOF configuration (same as TwistPolicy)
    obs_dof: DoFConfig = G1VisualmimicDoF()
    action_dof: DoFConfig = obs_dof  # Action DOF same as observation DOF

    # Tracker output configuration (from twist tracker)
    tracker_obs_total_degrees: int = 33  # Total degrees in tracker output
    tracker_obs_wrist_ids: list[int] = [27, 32]  # Indices of wrist DOFs in tracker output

    # Generator command configuration (generator outputs 31D tracker command)
    # Placeholder values - adjust based on actual trained model statistics
    generator_action_mean: list[float] = [0.0] * 31
    generator_action_std: list[float] = [1.0] * 31
    generator_clip_std_multiplier: float = 1.64

    # Tracker observation structure (same as Twist tracker setup)
    ankle_idx: list[int] = [4, 5, 10, 11]
    n_mimic_obs: int = 31

    # Multi-input support for generator model
    # PNP generator takes visual input (actor_obs_2d) and proprioceptive input (actor_obs)
    policy_input_keys: list[str] = ["actor_obs_2d", "actor_obs"]
    onnx_input_names: list[str] = ["actor_obs_2d", "actor_obs"]

    # Observation scales
    obs_scales: VisualmimicPolicyCfg.ObsScalesCfg = VisualmimicPolicyCfg.ObsScalesCfg(
        ang_vel=0.25, dof_vel=0.05, dof_pos=1.0
    )

    # Action post-processing
    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 1.0

    # Command mapping for high-level commands
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],  # forward/backward
        [1.0, 0.0, -1.0],  # left/right
        [1.0, 0.0, -1.0],  # turn
    ]
