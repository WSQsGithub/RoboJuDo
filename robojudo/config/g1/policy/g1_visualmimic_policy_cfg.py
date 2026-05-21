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

    # Default generator actions for history initialization.
    # Matches robot.control.generator.default_actions in pnp_config.yaml:
    # base_pos_z(1) + rpy(3) + base_lin_vel(3) + base_ang_vel_z(1) + dof_pos(23) = 31D
    generator_default_actions: list[float] = [
        0.793,                                          # base_pos_z
        0.0, 0.0, 0.0,                                  # rpy
        0.0, 0.0, 0.0,                                  # base_lin_vel
        0.0,                                            # base_ang_vel_z
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,               # left leg dof_pos
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,               # right leg dof_pos
        0.0, 0.0, 0.0,                                  # waist dof_pos
        0.0, 0.2, 0.0, 1.2,                             # left arm dof_pos
        0.0, -0.2, 0.0, 1.2,                            # right arm dof_pos
    ]

    # long_history mirrors obs_auxiliary.long_history in pnp_config.yaml
    long_history_config: dict[str, int] = {
        "generator_actions": 10,  # obs_auxiliary.long_history.generator_actions
        "tracker_proprio": 10,    # obs_auxiliary.long_history.tracker_proprio
    }

    # short_history mirrors obs_auxiliary.short_history in pnp_config.yaml
    short_history_config: dict[str, int] = {
        "commands": 5,
        "ee_pos_rel": 5,
        "ee_rot_rel": 5,
        "phase_one_hot": 5,
        "base_ang_vel": 5,
        "projected_gravity": 5,
        "dof_pos": 5,
        "dof_vel": 5,
        "actions": 5,
        "generator_actions": 5,
    }

    # Tracker observation structure (from obs_dict.tracker_obs in pnp_config.yaml)
    tracker_obs_config: list[str] = ["generator_actions", "tracker_proprio", "long_history"]

    # Actor observation structure (from obs_dict.actor_obs in pnp_config.yaml)
    actor_obs_config: list[str] = [
        "commands",
        "ee_pos_rel",
        "ee_rot_rel",
        "phase_one_hot",
        "base_ang_vel",
        "projected_gravity",
        "dof_pos",
        "dof_vel",
        "actions",
        "generator_actions",
        "short_history",
    ]

    # obs_dims from obs_dims in pnp_config.yaml (resolved values for G1 with 23 dofs, 31 generator dims)
    obs_dims: dict[str, int] = {
        "commands": 4,
        "ee_pos_rel": 6,
        "ee_rot_rel": 8,
        "phase_one_hot": 4,
        "base_ang_vel": 3,
        "base_rp": 2,
        "projected_gravity": 3,
        "dof_pos": 23,       # robot.dof_obs_size
        "dof_vel": 23,       # robot.dof_obs_size
        "actions": 23,       # robot.dof_obs_size
        "generator_actions": 31,  # robot.control.generator.dim_actions
        "tracker_proprio": 74,
    }

    # Observation scales from obs_scales in pnp_config.yaml
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

    # Tracker ankle DOF indices (zeroed in dof_vel for tracker_proprio)
    ankle_idx: list[int] = [4, 5, 10, 11]

    # Multi-input support for generator model
    # PNP generator takes visual input (actor_obs_2d) and proprioceptive input (actor_obs)
    policy_input_keys: list[str] = ["actor_obs_2d", "actor_obs"]
    onnx_input_names: list[str] = ["actor_obs_2d", "actor_obs"]

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
