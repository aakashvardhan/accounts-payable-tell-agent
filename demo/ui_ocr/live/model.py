"""Real model / Tell ports for the live worker (GPU). Loads Qwen3-8B ONCE plus the frozen safety LoRA, and the frozen probe.

  * The frozen adapter is hash-verified against results/lora_training/agent_s_v1/frozen_adapter/FROZEN.sha256 BEFORE it is attached,
    opened read-only (`is_trainable=False`), and never written.
  * The probe (weights, scaler, layer, threshold artifacts) is read from the frozen probe directory; its hashes are verified against
    FROZEN.sha256 there. The score is the same computation the project's own scripts use (standardise -> logistic).
  * Activation capture runs on the BASE weights (adapter disabled), exactly as the interface-demo/evaluation flow does.
All identifiers exposed to the UI come from these loaded artifacts, never from constants in the frontend.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from live import trusted as _t  # noqa: F401  (sys.path setup for tell.* / scripts)
from live.agent import Generation, Prepared, TellMeasurement
from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION
from tell.detector.capture import CaptureRequest, capture_predecision_activations, save_activation_artifact
from tell.safety.adapter_runtime import AdapterAwareRuntime

REPO = Path(__file__).resolve().parents[3]
FROZEN_ADAPTER = REPO / "results/lora_training/agent_s_v1/frozen_adapter"
FROZEN_PROBE = REPO / "results/probe_training/enterprise_v1/frozen_probe"
OPERATIONAL_ARTIFACT = REPO / "results/routing_design/operational_threshold_v1/operational_threshold_v1.json"
MAX_NEW_TOKENS = 1024     # evaluation protocol used 256; greedy decoding means a larger cap only lets long typed actions (evidence reports) finish


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen(directory: Path, must_list: tuple[str, ...]) -> dict:
    """Check every line of `<directory>/FROZEN.sha256`; raise if any file differs or a required file is unlisted. Returns {name: sha}."""
    listed = {}
    for line in (directory / "FROZEN.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip()] = digest
    for name in must_list:
        if name not in listed:
            raise RuntimeError(f"{directory.name}: {name} is not covered by FROZEN.sha256")
    for name, digest in listed.items():
        if sha_file(directory / name) != digest:
            raise RuntimeError(f"FROZEN artifact hash mismatch: {directory.name}/{name}")
    return listed


class RealPorts:
    """One instance = one model in memory. Implements both ModelPort and TellPort."""

    def __init__(self, activation_root: Path | None = None):
        self.adapter_hashes = verify_frozen(FROZEN_ADAPTER, ("adapter_model.safetensors", "adapter_config.json"))
        self.probe_hashes = verify_frozen(FROZEN_PROBE, ("probe_config.json", "probe_weights.safetensors"))
        cfg = json.loads((FROZEN_PROBE / "probe_config.json").read_text())
        self.layer = int(cfg["selected_layer"])
        self.probe_cfg = cfg
        w = load_file(str(FROZEN_PROBE / "probe_weights.safetensors"))
        self.scaler_mean, self.scaler_scale = w["scaler_mean"].astype(np.float64), w["scaler_scale"].astype(np.float64)
        self.coef, self.intercept = w["coef"].astype(np.float64), float(w["intercept"][0])
        t0 = time.perf_counter()
        self.rt = AdapterAwareRuntime()
        self.rt.load()
        self.adapter_info = self.rt.attach_agent_s_adapter(FROZEN_ADAPTER)
        if self.adapter_info.adapter_weights_sha256 != self.adapter_hashes["adapter_model.safetensors"]:
            raise RuntimeError("attached adapter hash differs from FROZEN.sha256")
        self.load_seconds = round(time.perf_counter() - t0, 1)
        self.activation_root = Path(activation_root) if activation_root else None
        self._ids = {
            "model_repo_id": PINNED_MODEL_REPO_ID, "model_revision": PINNED_MODEL_REVISION, "dtype": "bfloat16", "device": str(next(self.rt._runtime.model.parameters()).device),
            "adapter_weights_sha256": self.adapter_info.adapter_weights_sha256, "adapter_config_sha256": self.adapter_hashes["adapter_config.json"],
            "adapter_frozen_manifest_sha256": sha_file(FROZEN_ADAPTER / "FROZEN.sha256"), "adapter_dir": "results/lora_training/agent_s_v1/frozen_adapter",
            "probe_weights_sha256": self.probe_hashes["probe_weights.safetensors"], "probe_config_sha256": self.probe_hashes["probe_config.json"], "layer": self.layer,
            "token_position_rule": cfg.get("token_position_rule"), "scientific_threshold": cfg.get("threshold"),
            "operational_threshold_artifact_sha256": sha_file(OPERATIONAL_ARTIFACT), "load_seconds": self.load_seconds,
        }

    # ------------------------------------------------------------ shared
    def identifiers(self) -> dict:
        return dict(self._ids)

    # ------------------------------------------------------------ ModelPort
    def prepare(self, messages: list[dict]) -> Prepared:
        text = self.rt.render_chat_prompt(messages, enable_thinking=False)
        tok = self.rt.tokenize(text)
        return Prepared(text=text, tokens=tok, prompt_tokens=int(tok["input_ids"].shape[1]), prompt_sha256=hashlib.sha256(text.encode()).hexdigest())

    def generate(self, prepared: Prepared, agent: str) -> Generation:
        t0 = time.perf_counter()
        g = (self.rt.generate_as_agent_s if agent == "agent_s" else self.rt.generate_as_agent_1)(prepared.tokens, max_new_tokens=MAX_NEW_TOKENS)
        dt = time.perf_counter() - t0
        new = g[0][prepared.prompt_tokens:]
        return Generation(raw_output=self.rt.tokenizer.decode(new, skip_special_tokens=True), new_tokens=int(new.shape[0]), seconds=dt)

    # ------------------------------------------------------------ TellPort
    def measure(self, prepared: Prepared, *, run_id: str, turn: int) -> TellMeasurement:
        req = CaptureRequest(scenario_id=run_id, decision_point=f"live_turn_{turn}", source_ids=(run_id,), model_repo_id=PINNED_MODEL_REPO_ID, model_revision=PINNED_MODEL_REVISION,
                             tokenizer_class=type(self.rt.tokenizer).__name__, model_class="Qwen3ForCausalLM", prompt_text=prepared.text, hidden_state_indices=(self.layer,),
                             run_id=run_id, turn_number=turn)
        with self.rt.agent_1_model_context() as base:
            cap, vecs = capture_predecision_activations(base, prepared.tokens, req)
        vec = vecs[self.layer].numpy().astype(np.float64)
        z = (vec - self.scaler_mean) / self.scaler_scale
        logit = float(z @ self.coef + self.intercept)
        score = float(1.0 / (1.0 + math.exp(-logit)))
        artifact_sha = None
        if self.activation_root is not None:
            d = self.activation_root / run_id
            d.mkdir(parents=True, exist_ok=True)
            st, meta = d / f"turn{turn:02d}.safetensors", d / f"turn{turn:02d}_metadata.json"
            save_activation_artifact(vectors=vecs, capture_result=cap, request=req, safetensors_path=st, metadata_path=meta)
            artifact_sha = sha_file(st)
        return TellMeasurement(score=score, layer=self.layer, token_index=cap.selected_token_index, prompt_sha256=cap.prompt_sha256, input_ids_sha256=cap.input_ids_sha256,
                               elapsed_s=cap.elapsed_seconds, vector_norm=float(np.linalg.norm(vec)), dim=int(vec.shape[0]), artifact_sha256=artifact_sha)
