# 运行验证结果（单入口）

## ✅ 当前推荐入口

```bash
python scripts/run_pipeline.py -c g1_visualmimic
```

## ✅ 已验证内容

- 配置注册链路：`ConfigManager -> cfg_registry -> g1_visualmimic`
- 模型加载：`pnp_generator.onnx` + `twist_general_motion_tracker.pt`
- 训练配置读取：`assets/models/g1/visualmimic/pnp_config.yaml`
- 观测组装与 history 维护可正常进入运行循环

## 备注

- 运行中出现的 `QFontDatabase` 与仿真数值稳定性提示属于环境/控制参数问题，
  与入口脚本删除无直接关系。
