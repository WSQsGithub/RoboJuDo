# PNP 模型部署指南（单一路径）

当前仓库仅保留一条部署入口：`run_pipeline.py`。

## 文件位置

```
assets/models/g1/visualmimic/
├── pnp_config.yaml   # 训练导出的配置（运行时由 VisualmimicPolicy 直接读取）
└── pnp_generator.onnx

assets/models/g1/twist/
└── twist_general_motion_tracker.pt
```

## 运行方式

```bash
python scripts/run_pipeline.py -c g1_visualmimic
```

## 说明

- `G1VisualmimicPolicyCfg` 仅维护模型路径和训练配置路径。
- `VisualmimicPolicy` 初始化时直接读取 `assets/models/g1/visualmimic/pnp_config.yaml`。
- `obs_dict` / `obs_auxiliary` / `obs_dims` / `obs_scales` 等均来自该训练配置。

## 常见调整

- 替换模型：修改 `robojudo/config/g1/policy/g1_visualmimic_policy_cfg.py` 中的
  `generator_model_path` 和 `tracker_model_path`。
- 替换训练配置：修改同文件中的 `train_config_file`。
