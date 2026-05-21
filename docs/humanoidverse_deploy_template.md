# HumanoidVerse 部署模板（单入口）

当前仓库统一通过 `run_pipeline.py` 运行，不再提供独立脚本入口。

## 推荐运行

```bash
python scripts/run_pipeline.py -c g1_humanoidverse
```

## 视觉模仿（PNP）运行

```bash
python scripts/run_pipeline.py -c g1_visualmimic
```

## 关键点

- 策略配置放在 `robojudo/config/g1/policy/`。
- 运行时配置由 `ConfigManager` + `cfg_registry` 构建。
- 若模型路径或训练配置变更，直接修改对应 policy cfg。
