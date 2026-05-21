# PNP 模型部署指南

## 文件位置说明

你的文件已经放在正确的位置：

```
assets/models/g1/visualmimic/
├── pnp_config.yaml      # HumanoidVerse 导出的原始配置
├── pnp_model.onnx       # ONNX 模型文件
└── (转换后的部署配置自动生成在下面)

robojudo/deployment/humanoidverse/config/
└── pnp.yaml             # 自动转换生成的部署配置 ✓
```

## 使用步骤

### 1. 转换配置（已完成）

使用转换工具将 HumanoidVerse 导出的配置转换为部署配置：

```bash
cd /workspaces/RoboJuDo

python scripts/convert_humanoidverse_config.py \
  --humanoidverse-config assets/models/g1/visualmimic/pnp_config.yaml \
  --output robojudo/deployment/humanoidverse/config/pnp.yaml \
  --model-path assets/models/g1/visualmimic/pnp_model.onnx \
  --robot g1 \
  --backend onnx
```

**参数说明：**
- `--humanoidverse-config`: HumanoidVerse 导出的原始配置路径
- `--output`: 输出的部署配置文件位置
- `--model-path`: ONNX 模型文件路径
- `--robot`: 机器人类型（默认 g1）
- `--backend`: 推理后端，可选 `onnx`, `torchscript`, `dummy`

### 2. 运行模型推理

#### 方法 A：使用虚拟数据测试（推荐首先尝试）

```bash
cd /workspaces/RoboJuDo

python scripts/run_humanoidverse_deploy.py -cn pnp
```

此命令会：
- 加载 `pnp.yaml` 配置
- 使用虚拟随机观测数据
- 运行 5 步推理（可通过 `eval.steps=10` 修改）
- 输出每步的观测和动作维度

#### 方法 B：自定义参数运行

```bash
# 修改评估步数
python scripts/run_humanoidverse_deploy.py -cn pnp eval.steps=20

# 使用特定的执行提供者（加速）
python scripts/run_humanoidverse_deploy.py -cn pnp runtime.providers="[CUDAExecutionProvider,CPUExecutionProvider]"

# 修改模型路径
python scripts/run_humanoidverse_deploy.py -cn pnp runtime.model_path=/path/to/new/model.onnx
```

### 3. 集成到自己的代码

直接使用 `HumanoidVerseDeployer` 类进行推理：

```python
from omegaconf import OmegaConf
from robojudo.deployment.humanoidverse.deployer import HumanoidVerseDeployer

# 加载配置
cfg = OmegaConf.load("robojudo/deployment/humanoidverse/config/pnp.yaml")

# 创建部署器
deployer = HumanoidVerseDeployer(cfg)

# 准备观测数据（示例）
raw_obs = {
    "commands": np.array([0.5, 0.0, 0.1], dtype=np.float32),
    "ee_pos_rel": np.zeros(6, dtype=np.float32),
    # ... 其他观测项
}

# 执行一步推理
action, packed_obs = deployer.step(raw_obs)
print(f"Action shape: {action.shape}")
```

## 转换工具功能

### `scripts/convert_humanoidverse_config.py` 

这个工具自动做以下事情：

1. **读取 HumanoidVerse 配置**
   - 提取观测项 (`obs_terms`)
   - 提取观测组 (`obs_dict`)
   - 提取观测缩放 (`obs_scales`)
   - 提取历史配置 (`obs_auxiliary`)

2. **转换为部署格式**
   - 创建 `terms` 字典（包含维度和缩放因子）
   - 保留原始的观测组信息
   - 保持历史配置
   - 设置运行时参数（后端、模型路径等）

3. **生成可复用的配置文件**
   - YAML 格式，易于手动修改
   - 继承 `base.yaml` 的基础配置
   - 所有 HumanoidVerse 特定的信息都已提取

### 转换后配置的关键部分

```yaml
# 基础设置
defaults:
  - base  # 继承基础配置

# 运行时配置
runtime:
  backend: onnx
  model_path: assets/models/g1/visualmimic/pnp_model.onnx
  action_dim: 23  # 从模型推断

# 观测配置
obs:
  clip: 80.0  # 从环保规范化设置
  terms:     # 所有观测项的维度和缩放
    commands:
      dim: 4
      scale: 1.0
    ...
  groups:    # 观测组定义
    actor_obs:
      - commands
      - ...
  history:   # 历史观测配置
    short_history:
      commands: 5
      ...
```

## 常见问题

### Q: 如何修改模型路径？

编辑 `robojudo/deployment/humanoidverse/config/pnp.yaml` 中的 `runtime.model_path`，或运行时覆盖：

```bash
python scripts/run_humanoidverse_deploy.py -cn pnp runtime.model_path=/new/path/model.onnx
```

### Q: 如何使用 GPU 推理？

修改 `runtime.providers`：

```bash
python scripts/run_humanoidverse_deploy.py -cn pnp 'runtime.providers=[CUDAExecutionProvider,CPUExecutionProvider]'
```

### Q: 为什么要用这个工具而不是手工编辑？

因为：
- 自动处理所有观测项和维度映射
- 避免手工出错
- 可以快速适配新的模型导出
- 保持一致性

## 下一步

1. ✅ 已转换配置文件
2. 🔄 测试推理（需要依赖已安装）
3. 📊 集成到机器人控制系统
4. 🚀 部署到机器人硬件
