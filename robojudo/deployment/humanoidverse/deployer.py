from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from robojudo.deployment.humanoidverse.obs_packer import AutoObsPacker


class HumanoidVerseDeployer:
    def __init__(self, cfg: DictConfig):
        self.cfg = self._normalize_config(cfg)
        self.obs_packer = AutoObsPacker(self.cfg.obs)
        self._policy_input_key = str(self.cfg.runtime.get("policy_input_key", "policy_input"))
        self._policy_input_keys = [str(x) for x in self.cfg.runtime.get("policy_input_keys", [])]
        self._action_dim = int(self.cfg.runtime.action_dim)
        self._backend = str(self.cfg.runtime.backend)
        self._input_name = str(self.cfg.runtime.get("input_name", self._policy_input_key))
        self._input_names = [str(x) for x in self.cfg.runtime.get("input_names", [])]
        self._input_shapes = {
            str(key): [int(v) for v in value]
            for key, value in self.cfg.runtime.get("input_shapes", {}).items()
        }

        self._session = None
        self._onnx_input_names: list[str] = []
        self._onnx_output_name = None
        self._torch_model = None

        self._init_runtime()

    def _normalize_config(self, cfg: DictConfig) -> DictConfig:
        if "runtime" in cfg and "obs" in cfg and "terms" in cfg.obs and "outputs" in cfg.obs:
            return cfg

        if "obs" not in cfg or "obs_dict" not in cfg.obs:
            raise ValueError("Unsupported deploy config: expected template obs format or humanoidverse exported obs format")

        output_key = "actor_obs"
        if "runtime" in cfg and "policy_input_key" in cfg.runtime:
            output_key = str(cfg.runtime.policy_input_key)

        obs_dims_cfg = cfg.obs.get("obs_dims", [])
        obs_dims: dict[str, int] = {}
        for item in obs_dims_cfg:
            for key, value in item.items():
                obs_dims[str(key)] = int(value)

        obs_scales_cfg = cfg.obs.get("obs_scales", {})
        obs_scales = {str(key): float(value) for key, value in obs_scales_cfg.items()}

        terms: dict[str, dict[str, float | int]] = {}
        for term_name, dim in obs_dims.items():
            terms[term_name] = {
                "dim": dim,
                "scale": obs_scales.get(term_name, 1.0),
            }

        groups = {str(group_name): [str(token) for token in tokens] for group_name, tokens in cfg.obs.obs_dict.items()}

        history: dict[str, dict[str, Any]] = {}
        for aux_name, aux_spec in cfg.obs.get("obs_auxiliary", {}).items():
            history[str(aux_name)] = {
                "terms": {str(term_name): int(repeat_steps) for term_name, repeat_steps in aux_spec.items()}
            }

        outputs: dict[str, list[str]] = {}
        if output_key in groups:
            outputs[output_key] = [output_key]
        else:
            outputs[output_key] = ["actor_obs"] if "actor_obs" in groups else list(groups.keys())[:1]

        clip_obs = 100.0
        if "env" in cfg and "config" in cfg.env and "normalization" in cfg.env.config:
            clip_obs = float(cfg.env.config.normalization.get("clip_observations", 100.0))

        action_dim = None
        if "runtime" in cfg and "action_dim" in cfg.runtime:
            action_dim = int(cfg.runtime.action_dim)
        elif "robot" in cfg:
            if "actions_dim" in cfg.robot:
                action_dim = int(cfg.robot.actions_dim)
            elif "number_of_actions" in cfg.robot and cfg.robot.number_of_actions != "???":
                action_dim = int(cfg.robot.number_of_actions)

        normalized = {
            "runtime": {
                "backend": cfg.get("runtime", {}).get("backend", "dummy"),
                "model_path": cfg.get("runtime", {}).get("model_path", None),
                "providers": cfg.get("runtime", {}).get("providers", ["CPUExecutionProvider"]),
                "input_name": cfg.get("runtime", {}).get("input_name", output_key),
                "input_names": cfg.get("runtime", {}).get("input_names", []),
                "input_shapes": cfg.get("runtime", {}).get("input_shapes", {}),
                "output_name": cfg.get("runtime", {}).get("output_name", "action"),
                "policy_input_key": output_key,
                "policy_input_keys": cfg.get("runtime", {}).get("policy_input_keys", []),
                "action_dim": action_dim if action_dim is not None else 0,
            },
            "obs": {
                "clip": clip_obs,
                "terms": terms,
                "groups": groups,
                "history": history,
                "outputs": outputs,
            },
        }

        if normalized["runtime"]["action_dim"] <= 0:
            raise ValueError("Unable to infer runtime.action_dim from config; please provide runtime.action_dim")

        return OmegaConf.create(normalized)

    def _init_runtime(self) -> None:
        if self._backend == "dummy":
            return

        model_path = Path(str(self.cfg.runtime.model_path)).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        if self._backend == "onnx":
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(model_path), providers=list(self.cfg.runtime.get("providers", ["CPUExecutionProvider"]))
            )
            self._onnx_input_names = [inp.name for inp in self._session.get_inputs()]
            self._onnx_output_name = str(self.cfg.runtime.get("output_name", self._session.get_outputs()[0].name))
            return

        if self._backend == "torchscript":
            import torch

            self._torch_model = torch.jit.load(str(model_path), map_location="cpu")
            self._torch_model.eval()
            return

        raise ValueError(f"unsupported runtime backend '{self._backend}'")

    def _resolve_policy_input_keys(self, packed: dict[str, np.ndarray]) -> list[str]:
        if self._policy_input_keys:
            missing = [key for key in self._policy_input_keys if key not in packed]
            if missing:
                raise KeyError(f"Configured policy_input_keys missing from packed outputs: {missing}")
            return list(self._policy_input_keys)

        if self._policy_input_key in packed:
            return [self._policy_input_key]

        actor_like = [key for key in packed.keys() if key.startswith("actor_obs")]
        if actor_like:
            actor_like.sort(key=lambda x: (x != "actor_obs", x))
            return actor_like

        if len(packed) == 1:
            return [next(iter(packed.keys()))]

        raise KeyError(
            f"policy input key '{self._policy_input_key}' not found in packed outputs: {list(packed.keys())}"
        )

    def _reshape_input_if_needed(self, input_key: str, obs_vec: np.ndarray) -> np.ndarray:
        if input_key in self._input_shapes:
            target_shape = tuple(self._input_shapes[input_key])
            return obs_vec.reshape(target_shape)
        return obs_vec

    def infer(self, obs_inputs: dict[str, np.ndarray]) -> np.ndarray:
        if self._backend == "dummy":
            return np.zeros((self._action_dim,), dtype=np.float32)

        if self._backend == "onnx":
            assert self._session is not None
            assert self._onnx_output_name is not None

            input_keys = list(obs_inputs.keys())
            if self._input_names:
                onnx_names = list(self._input_names)
            elif len(input_keys) == 1:
                onnx_names = [self._input_name]
            elif all(key in self._onnx_input_names for key in input_keys):
                onnx_names = input_keys
            else:
                if len(input_keys) != len(self._onnx_input_names):
                    raise ValueError(
                        f"Cannot map inputs automatically: obs_keys={input_keys}, onnx_inputs={self._onnx_input_names}. "
                        "Please set runtime.input_names and runtime.policy_input_keys explicitly."
                    )
                onnx_names = list(self._onnx_input_names)

            if len(onnx_names) != len(input_keys):
                raise ValueError(
                    f"Input mapping mismatch: input_keys={input_keys}, onnx_names={onnx_names}."
                )

            feed_dict: dict[str, np.ndarray] = {}
            for onnx_name, input_key in zip(onnx_names, input_keys, strict=True):
                obs_vec = obs_inputs[input_key].astype(np.float32, copy=False)
                obs_vec = self._reshape_input_if_needed(input_key, obs_vec)
                feed_dict[onnx_name] = np.expand_dims(obs_vec, axis=0)

            outputs = self._session.run([self._onnx_output_name], feed_dict)
            return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        assert self._torch_model is not None
        import torch

        if len(obs_inputs) != 1:
            raise ValueError("torchscript backend currently supports single policy input only")

        input_key = next(iter(obs_inputs.keys()))
        obs_vec = self._reshape_input_if_needed(input_key, obs_inputs[input_key])

        if np.asarray(obs_vec).ndim != 1:
            raise ValueError("torchscript backend expects 1D input vector; consider using onnx backend for multi/head inputs")

        obs_tensor = torch.from_numpy(obs_vec.astype(np.float32, copy=False)).unsqueeze(0)
        with torch.inference_mode():
            action = self._torch_model(obs_tensor).cpu().numpy().reshape(-1)
        return action.astype(np.float32, copy=False)

    def step(self, raw_obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        packed = self.obs_packer.pack(raw_obs)
        input_keys = self._resolve_policy_input_keys(packed)
        input_obs = {key: packed[key] for key in input_keys}

        action = self.infer(input_obs)
        if action.shape[0] != self._action_dim:
            raise ValueError(f"action dim mismatch: expected {self._action_dim}, got {action.shape[0]}")
        return action, packed
