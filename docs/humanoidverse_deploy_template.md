# HumanoidVerse 部署模板

新增模板位于 `robojudo/deployment/humanoidverse/`，包含：

- `deployer.py`：统一部署入口，支持 `dummy` / `onnx` / `torchscript` 后端。
- `obs_packer.py`：自动 obs 组包器，按配置完成项拼接、缩放、裁剪、history 组装。
- `config/base.yaml`：Hydra 配置模板。
- `scripts/run_humanoidverse_deploy.py`：Hydra 入口示例。

## 配置要点

`obs` 配置分为四层：

- `terms`：原子观测项（维度、缩放系数）。
- `groups`：将若干 `terms` 或其他 `groups` 拼接成中间向量。
- `history`：对某个 `group` 自动维护时序堆叠。
- `outputs`：最终输出给模型的观测键。

默认模板中 `policy_input = actor_obs + actor_history`。

## 直接使用 HumanoidVerse 导出 `config.yaml`

现在支持直接读取 HumanoidVerse 导出配置（例如 `.hydra/config.yaml`），会自动提取部署所需字段。

核心读取字段（重点是 obs 组成）：

- `obs.obs_dict`：定义输出观测组（如 `actor_obs`、`critic_obs`）及拼接顺序。
- `obs.obs_dims`：每个原子观测项的维度（支持 list[dict] 格式）。
- `obs.obs_scales`：每个原子观测项的缩放。
- `obs.obs_auxiliary`：辅助历史观测（如 `history_long`），自动转为 history 组包规则。
- `env.config.normalization.clip_observations`：观测裁剪阈值。
- `robot.actions_dim`（或 `robot.number_of_actions`）：动作维度。

对你提供的样例，`actor_obs` 将按如下顺序组包：

`base_lin_vel + base_ang_vel + projected_gravity + command_lin_vel + command_ang_vel + dof_pos + dof_vel + actions`

其中维度由 `obs.obs_dims` 决定，缩放由 `obs.obs_scales` 决定。

## 快速运行

```bash
python scripts/run_humanoidverse_deploy.py
```

默认使用 `dummy` 后端，便于先验证 obs 组包链路。

## 与现有部署链路一致（推荐）

现在已接入 RoboJuDo 原生配置体系，可直接通过：

```bash
python scripts/run_pipeline.py -c g1_humanoidverse
```

该配置走标准 `ConfigManager -> RlPipeline -> Policy` 路径，和仓库其它部署逻辑一致。

对应配置入口：

- `robojudo/config/g1/g1_cfg.py` 中的 `g1_humanoidverse`
- `robojudo/config/g1/policy/g1_humanoidverse_policy_cfg.py`
- `robojudo/policy/humanoidverse_policy.py`

## 切到 ONNX 部署

```bash
python scripts/run_humanoidverse_deploy.py \
  runtime.backend=onnx \
  runtime.model_path=/path/to/policy.onnx \
  runtime.input_name=actor_obs \
  runtime.output_name=action \
  runtime.action_dim=12
```

如果模型输入不是 `policy_input`，可通过：

```bash
python scripts/run_humanoidverse_deploy.py runtime.policy_input_key=your_obs_key
```

如果使用 `run_pipeline.py -c g1_humanoidverse`，建议在
`G1HumanoidVersePolicyCfg` 中设置：

- `runtime_backend="onnx"`
- `model_path="/path/to/policy.onnx"`
- 需要时增加 `hydra_overrides=[...]`
