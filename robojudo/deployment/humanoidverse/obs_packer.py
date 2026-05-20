from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class ObsTerm:
    name: str
    dim: int
    scale: float


class AutoObsPacker:
    def __init__(self, obs_cfg: DictConfig):
        self._clip = float(obs_cfg.get("clip", 100.0))
        self._terms = self._parse_terms(obs_cfg.terms)
        self._groups = {key: list(value) for key, value in obs_cfg.groups.items()}
        self._history_cfg: dict[str, dict[str, Any]] = {}
        for key, value in obs_cfg.get("history", {}).items():
            if "terms" in value:
                self._history_cfg[key] = {
                    "terms": {term_name: int(repeat_steps) for term_name, repeat_steps in value.terms.items()}
                }
            else:
                self._history_cfg[key] = {
                    "source": str(value.source),
                    "steps": int(value.steps),
                }
        self._outputs = {key: list(value) for key, value in obs_cfg.outputs.items()}
        self._history_buf: dict[str, Any] = {}

        self._validate_static_config()

    def term_names(self) -> list[str]:
        return list(self._terms.keys())

    @staticmethod
    def _parse_terms(terms_cfg: DictConfig) -> dict[str, ObsTerm]:
        terms: dict[str, ObsTerm] = {}
        for key, value in terms_cfg.items():
            terms[key] = ObsTerm(name=key, dim=int(value.dim), scale=float(value.get("scale", 1.0)))
        return terms

    def _validate_static_config(self) -> None:
        for term in self._terms.values():
            if term.dim <= 0:
                raise ValueError(f"obs term '{term.name}' has invalid dim={term.dim}")

        for group_name, tokens in self._groups.items():
            if not tokens:
                raise ValueError(f"obs group '{group_name}' is empty")
            for token in tokens:
                self._validate_token(token, scope=group_name)

        for hist_name, hist_cfg in self._history_cfg.items():
            if "source" in hist_cfg:
                if hist_cfg["steps"] <= 0:
                    raise ValueError(f"history '{hist_name}' has invalid steps={hist_cfg['steps']}")
                source = hist_cfg["source"]
                if source not in self._groups:
                    raise ValueError(f"history '{hist_name}' references unknown source group '{source}'")
                continue

            if "terms" in hist_cfg:
                terms = hist_cfg["terms"]
                if not terms:
                    raise ValueError(f"history '{hist_name}' has empty terms")
                for term_name, repeat_steps in terms.items():
                    if repeat_steps <= 0:
                        raise ValueError(
                            f"history '{hist_name}' term '{term_name}' has invalid steps={repeat_steps}"
                        )
                    self._validate_token(term_name, scope=hist_name)
                continue

            raise ValueError(f"history '{hist_name}' must define either source/steps or terms")

        for out_name, tokens in self._outputs.items():
            if not tokens:
                raise ValueError(f"obs output '{out_name}' is empty")
            for token in tokens:
                if token not in self._groups and token not in self._history_cfg and token not in self._terms:
                    raise ValueError(f"obs output '{out_name}' references unknown token '{token}'")

    def _validate_token(self, token: str, scope: str) -> None:
        base = token[:-4] if token.endswith("_raw") else token
        if base in self._terms or base in self._groups:
            return
        raise ValueError(f"obs group '{scope}' references unknown token '{token}'")

    def _fetch_term(self, term_name: str, raw_obs: dict[str, Any]) -> np.ndarray:
        if term_name not in raw_obs:
            raise KeyError(f"missing observation term '{term_name}' in raw_obs")
        term_cfg = self._terms[term_name]
        arr = np.asarray(raw_obs[term_name], dtype=np.float32).reshape(-1)
        if arr.shape[0] != term_cfg.dim:
            raise ValueError(
                f"observation term '{term_name}' dim mismatch: expected {term_cfg.dim}, got {arr.shape[0]}"
            )
        return arr

    def _compose(self, name: str, raw_obs: dict[str, Any], cache: dict[str, np.ndarray]) -> np.ndarray:
        if name in cache:
            return cache[name]

        if name in self._terms:
            term = self._fetch_term(name, raw_obs)
            scaled = np.clip(term * self._terms[name].scale, -self._clip, self._clip)
            cache[name] = scaled
            return scaled

        if name not in self._groups:
            raise KeyError(f"unknown compose target '{name}'")

        pieces: list[np.ndarray] = []
        for token in self._groups[name]:
            raw_mode = token.endswith("_raw")
            base = token[:-4] if raw_mode else token
            if base in self._terms:
                term = self._fetch_term(base, raw_obs)
                if not raw_mode:
                    term = np.clip(term * self._terms[base].scale, -self._clip, self._clip)
                pieces.append(term)
            else:
                pieces.append(self._compose(base, raw_obs, cache))

        packed = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        cache[name] = packed
        return packed

    def _compose_history(self, hist_name: str, source: np.ndarray) -> np.ndarray:
        hist_cfg = self._history_cfg[hist_name]
        steps = hist_cfg["steps"]

        if hist_name not in self._history_buf:
            self._history_buf[hist_name] = deque(maxlen=steps)

        buf = self._history_buf[hist_name]
        if not buf:
            for _ in range(steps):
                buf.append(source.copy())
        else:
            buf.append(source.copy())

        return np.concatenate(list(buf), axis=0).astype(np.float32, copy=False)

    def _compose_history_terms(
        self,
        hist_name: str,
        terms_cfg: dict[str, int],
        raw_obs: dict[str, Any],
        cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        if hist_name not in self._history_buf:
            self._history_buf[hist_name] = {}

        hist_buf_by_term: dict[str, deque[np.ndarray]] = self._history_buf[hist_name]
        pieces: list[np.ndarray] = []

        for token, steps in terms_cfg.items():
            raw_mode = token.endswith("_raw")
            base = token[:-4] if raw_mode else token

            if base in self._terms:
                source_vec = self._fetch_term(base, raw_obs)
                if not raw_mode:
                    source_vec = np.clip(source_vec * self._terms[base].scale, -self._clip, self._clip)
            elif base in self._groups:
                source_vec = self._compose(base, raw_obs, cache)
            else:
                raise KeyError(f"unknown history token '{token}' in history '{hist_name}'")

            if token not in hist_buf_by_term:
                hist_buf_by_term[token] = deque(maxlen=steps)

            buf = hist_buf_by_term[token]
            if not buf:
                for _ in range(steps):
                    buf.append(source_vec.copy())
            else:
                buf.append(source_vec.copy())

            pieces.append(np.concatenate(list(buf), axis=0).astype(np.float32, copy=False))

        return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

    def pack(self, raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
        cache: dict[str, np.ndarray] = {}

        for term_name in self._terms:
            self._compose(term_name, raw_obs, cache)

        for group_name in self._groups:
            self._compose(group_name, raw_obs, cache)

        for hist_name, hist_cfg in self._history_cfg.items():
            if "source" in hist_cfg:
                source_vec = cache[hist_cfg["source"]]
                cache[hist_name] = self._compose_history(hist_name, source_vec)
            else:
                cache[hist_name] = self._compose_history_terms(hist_name, hist_cfg["terms"], raw_obs, cache)

        outputs: dict[str, np.ndarray] = {}
        for out_name, tokens in self._outputs.items():
            pieces = []
            for token in tokens:
                if token in cache:
                    pieces.append(cache[token])
                elif token in self._terms:
                    pieces.append(self._compose(token, raw_obs, cache))
                else:
                    raise KeyError(f"unknown output token '{token}' in output '{out_name}'")
            outputs[out_name] = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

        return outputs

    def expected_dim(self, output_name: str) -> int:
        if output_name not in self._outputs:
            raise KeyError(f"unknown output name '{output_name}'")
        dim = 0
        for token in self._outputs[output_name]:
            if token in self._terms:
                dim += self._terms[token].dim
            elif token in self._groups:
                dim += self._group_dim(token)
            elif token in self._history_cfg:
                hist_cfg = self._history_cfg[token]
                if "source" in hist_cfg:
                    src = hist_cfg["source"]
                    dim += self._group_dim(src) * hist_cfg["steps"]
                else:
                    for term_name, repeat_steps in hist_cfg["terms"].items():
                        base = term_name[:-4] if term_name.endswith("_raw") else term_name
                        if base in self._terms:
                            dim += self._terms[base].dim * repeat_steps
                        else:
                            dim += self._group_dim(base) * repeat_steps
            else:
                raise KeyError(f"unknown token '{token}' in output '{output_name}'")
        return dim

    def _group_dim(self, group_name: str) -> int:
        dim = 0
        for token in self._groups[group_name]:
            base = token[:-4] if token.endswith("_raw") else token
            if base in self._terms:
                dim += self._terms[base].dim
            else:
                dim += self._group_dim(base)
        return dim
