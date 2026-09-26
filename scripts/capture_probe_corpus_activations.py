"""Forward-only activation capture for the probe corpus (Section 12-13).

Loads Qwen3-8B once. For each of the 200 samples in
`results/probe_dataset/rendered_samples.jsonl`, renders the frozen
messages, tokenizes once, and runs exactly one forward pass
(`output_hidden_states=True, use_cache=False, torch.inference_mode()`)
via the existing, unmodified `tell.detector.capture
.capture_predecision_activations` / `save_activation_artifact` -- no
generation, no model action, ever. Only hidden-state indices 9, 18, 27,
36 are requested (embedding index 0 is excluded, per the frozen
protocol manifest).

Resumable, atomic per-sample capture: each sample's safetensors +
metadata JSON is first written to a `.tmp` path under
`results/probe_dataset/activations/_tmp_per_sample/` and only renamed to
its final name after both writes succeed. A rerun scans this directory
for already-completed `{sample_id}.safetensors` + `{sample_id}_metadata
.json` pairs and skips them, so an interrupted run loses at most the one
sample in flight when it was interrupted.

After every sample has a completed per-sample artifact, this script
consolidates them into the compact, training-ready sharded format
(Section 13): one SafeTensors file per (split, layer) shaped
`[n_samples_in_split, 4096]`, plus a sample-index mapping and a single
consolidated capture-metadata JSONL (no labels in either). The
per-sample temporary directory is deleted only after the consolidated
shards are re-read back and verified against the per-sample sources.

No generation occurs anywhere in this script; no `PayInvoiceCandidate`,
gate, or ledger/SQLite operation is constructed.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tell.agent.local_model import PINNED_MODEL_REPO_ID, PINNED_MODEL_REVISION, PINNED_SNAPSHOT_PATH, QwenLocalRuntime
from tell.detector.capture import CaptureRequest, capture_predecision_activations, load_activation_artifact, save_activation_artifact

OUTPUT_DIR = Path("/home/hp5/tell/results/probe_dataset")
RENDERED_SAMPLES_PATH = OUTPUT_DIR / "rendered_samples.jsonl"
LABELS_PATH = OUTPUT_DIR / "labels.jsonl"
PROTOCOL_MANIFEST_PATH = OUTPUT_DIR / "corpus_protocol_manifest.json"

ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
TMP_PER_SAMPLE_DIR = ACTIVATIONS_DIR / "_tmp_per_sample"
SAMPLE_INDEX_MAPPING_PATH = ACTIVATIONS_DIR / "sample_index_mapping.json"
CAPTURE_METADATA_PATH = ACTIVATIONS_DIR / "capture_metadata.jsonl"
CAPTURE_REPORT_PATH = OUTPUT_DIR / "capture_report.md"

HIDDEN_STATE_INDICES = (9, 18, 27, 36)
HIDDEN_SIZE = 4096


def _load_rendered_samples() -> list[dict]:
    records = []
    with RENDERED_SAMPLES_PATH.open() as f:
        for line in f:
            records.append(json.loads(line))
    records.sort(key=lambda r: r["sample_id"])  # deterministic order
    return records


def _load_split_by_sample_id() -> dict[str, str]:
    mapping = {}
    with LABELS_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["sample_id"]] = rec["split"]
    return mapping


def _sample_done(sample_id: str) -> bool:
    st = TMP_PER_SAMPLE_DIR / f"{sample_id}.safetensors"
    md = TMP_PER_SAMPLE_DIR / f"{sample_id}_metadata.json"
    return st.exists() and md.exists()


def _capture_one(runtime: QwenLocalRuntime, record: dict) -> dict:
    chat_text = runtime.render_chat_prompt(record["messages"], enable_thinking=False)
    inputs = runtime.tokenize(chat_text)

    capture_request = CaptureRequest(
        scenario_id=record["sample_id"],
        decision_point=record["decision_point"],
        source_ids=(record["docid"],),
        model_repo_id=PINNED_MODEL_REPO_ID,
        model_revision=PINNED_MODEL_REVISION,
        tokenizer_class=type(runtime.tokenizer).__name__,
        model_class=type(runtime.model).__name__,
        prompt_text=chat_text,
        hidden_state_indices=HIDDEN_STATE_INDICES,
    )
    capture_result, vectors = capture_predecision_activations(runtime.model, inputs, capture_request)

    for idx, vec in vectors.items():
        if not bool(torch.isfinite(vec).all().item()):
            raise RuntimeError(f"{record['sample_id']}: non-finite values in hidden_state_{idx}")

    tmp_st = TMP_PER_SAMPLE_DIR / f"{record['sample_id']}.safetensors.tmp"
    tmp_md = TMP_PER_SAMPLE_DIR / f"{record['sample_id']}_metadata.json.tmp"
    final_st = TMP_PER_SAMPLE_DIR / f"{record['sample_id']}.safetensors"
    final_md = TMP_PER_SAMPLE_DIR / f"{record['sample_id']}_metadata.json"

    save_activation_artifact(vectors=vectors, capture_result=capture_result, request=capture_request, safetensors_path=tmp_st, metadata_path=tmp_md)
    tmp_st.rename(final_st)
    tmp_md.rename(final_md)

    return {
        "sample_id": record["sample_id"],
        "capture_elapsed_seconds": capture_result.elapsed_seconds,
        "sequence_length": capture_result.sequence_length,
        "peak_memory_allocated_bytes": capture_result.peak_memory_allocated_bytes,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL_MANIFEST_PATH.read_text())
    assert tuple(protocol["capture_hidden_state_indices"]) == HIDDEN_STATE_INDICES

    records = _load_rendered_samples()
    if len(records) != 200:
        raise RuntimeError(f"Expected 200 rendered samples, found {len(records)}")
    split_by_id = _load_split_by_sample_id()

    TMP_PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up any half-written .tmp leftovers from a prior interrupted run --
    # never touches a completed {sample_id}.safetensors/_metadata.json pair.
    for leftover in TMP_PER_SAMPLE_DIR.glob("*.tmp"):
        leftover.unlink()

    pending = [r for r in records if not _sample_done(r["sample_id"])]
    already_done = len(records) - len(pending)
    print(f"[capture] {already_done}/{len(records)} samples already captured; {len(pending)} pending")

    load_result = None
    timing_records = []
    if pending:
        runtime = QwenLocalRuntime(PINNED_SNAPSHOT_PATH)
        load_result = runtime.load()
        print(f"[load] {load_result.elapsed_seconds:.2f}s, peak allocated {load_result.peak_memory_allocated_bytes / 1e9:.2f} GB")

        for i, record in enumerate(pending, start=1):
            timing = _capture_one(runtime, record)
            timing_records.append(timing)
            if i % 20 == 0 or i == len(pending):
                print(f"[capture] {i}/{len(pending)} ({record['sample_id']}) elapsed={timing['capture_elapsed_seconds']:.3f}s tokens={timing['sequence_length']}")

        runtime.unload()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[capture] nothing pending -- all samples already captured")

    # ---- Consolidation: per-sample tmp artifacts -> compact per-split-per-layer shards ----
    print("[consolidate] building per-split-per-layer shards ...")
    sample_index_mapping: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    per_split_vectors: dict[str, dict[int, list[torch.Tensor]]] = {
        split: {idx: [] for idx in HIDDEN_STATE_INDICES} for split in ("train", "validation", "test")
    }
    capture_metadata_records = []

    for record in records:  # deterministic sample_id-sorted order
        sample_id = record["sample_id"]
        split = split_by_id[sample_id]
        st_path = TMP_PER_SAMPLE_DIR / f"{sample_id}.safetensors"
        md_path = TMP_PER_SAMPLE_DIR / f"{sample_id}_metadata.json"
        vectors = load_activation_artifact(st_path)
        metadata = json.loads(md_path.read_text())

        for idx in HIDDEN_STATE_INDICES:
            vec = vectors[idx]
            if vec.shape != (HIDDEN_SIZE,):
                raise RuntimeError(f"{sample_id}: hidden_state_{idx} has shape {tuple(vec.shape)}, expected ({HIDDEN_SIZE},)")
            per_split_vectors[split][idx].append(vec)
        sample_index_mapping[split].append(sample_id)

        capture_metadata_records.append(
            {
                "sample_id": sample_id,
                "docid": record["docid"],
                "decision_point": record["decision_point"],
                "split": split,
                "prompt_sha256": metadata["prompt_sha256"],
                "input_ids_sha256": metadata["input_ids_sha256"],
                "sequence_length": metadata["sequence_length"],
                "selected_token_index": metadata["selected_token_index"],
                "selected_token_id": metadata["selected_token_id"],
                "hidden_state_indices": metadata["hidden_state_indices"],
                "activations": metadata["activations"],
                "timing": metadata["timing"],
            }
        )

    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        for idx in HIDDEN_STATE_INDICES:
            stacked = torch.stack(per_split_vectors[split][idx], dim=0).contiguous()  # [n_split, 4096]
            final_path = ACTIVATIONS_DIR / f"{split}_layer{idx}.safetensors"
            tmp_path = ACTIVATIONS_DIR / f"{split}_layer{idx}.safetensors.tmp"
            save_file({"activations": stacked}, str(tmp_path), metadata={"split": split, "hidden_state_index": str(idx), "n_samples": str(stacked.shape[0])})
            tmp_path.rename(final_path)

    SAMPLE_INDEX_MAPPING_PATH.write_text(json.dumps(sample_index_mapping, indent=2))
    with CAPTURE_METADATA_PATH.open("w") as f:
        for r in capture_metadata_records:
            f.write(json.dumps(r) + "\n")

    print(f"[consolidate] wrote shards for splits: { {s: len(v) for s, v in sample_index_mapping.items()} }")

    # ---- Verify consolidated shards against per-sample sources before cleanup ----
    ok = True
    for split in ("train", "validation", "test"):
        for idx in HIDDEN_STATE_INDICES:
            from safetensors import safe_open

            with safe_open(str(ACTIVATIONS_DIR / f"{split}_layer{idx}.safetensors"), framework="pt") as f:
                shard = f.get_tensor("activations")
            for row, sample_id in enumerate(sample_index_mapping[split]):
                original = load_activation_artifact(TMP_PER_SAMPLE_DIR / f"{sample_id}.safetensors")[idx]
                if not torch.equal(shard[row], original):
                    ok = False
                    print(f"[verify] MISMATCH split={split} layer={idx} row={row} sample_id={sample_id}")
    print(f"[verify] consolidated shards match per-sample sources exactly: {ok}")
    if not ok:
        raise RuntimeError("Consolidated shard verification failed -- not deleting per-sample tmp artifacts.")

    for f in TMP_PER_SAMPLE_DIR.glob("*"):
        f.unlink()
    TMP_PER_SAMPLE_DIR.rmdir()
    print(f"[cleanup] removed {TMP_PER_SAMPLE_DIR} after successful consolidation")

    if load_result is not None:
        cleanup_mem = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
        lines = ["# Probe Corpus Capture Report\n"]
        lines.append(f"Captured {len(pending)} new samples this run ({already_done} were already captured from a prior run).\n")
        lines.append(f"Model load: {load_result.elapsed_seconds:.2f}s, peak allocated {load_result.peak_memory_allocated_bytes/1e9:.2f} GB.\n")
        total_capture_time = sum(t["capture_elapsed_seconds"] for t in timing_records)
        lines.append(f"Total forward-pass capture time this run: {total_capture_time:.2f}s over {len(timing_records)} samples "
                     f"(mean {total_capture_time/max(len(timing_records),1):.3f}s/sample).\n")
        lines.append(f"Cleanup after capture: CUDA allocated {cleanup_mem['allocated_bytes']/1e6:.2f} MB, reserved {cleanup_mem['reserved_bytes']/1e6:.2f} MB.\n")
        lines.append(f"Consolidated shard verification: **{ok}**.\n")
        lines.append("No generation, no model action, no PayInvoiceCandidate/gate/ledger/SQLite operation occurred in this script.\n")
        CAPTURE_REPORT_PATH.write_text("\n".join(lines))
        print(f"Wrote {CAPTURE_REPORT_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
