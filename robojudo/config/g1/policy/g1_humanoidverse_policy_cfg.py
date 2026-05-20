from robojudo.policy.policy_cfgs import HumanoidVersePolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from .g1_unitree_policy_cfg import G1UnitreeDoF


class G1HumanoidVersePolicyCfg(HumanoidVersePolicyCfg):
    robot: str = "g1"

    runtime_backend: str = "dummy"
    model_path: str | None = None

    obs_dof: DoFConfig = G1UnitreeDoF()
    action_dof: DoFConfig = obs_dof
