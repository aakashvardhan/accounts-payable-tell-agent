# Tell — activation-guided security for an accounts-payable agent

Tell protects an autonomous accounts-payable (AP) agent from prompt injection and memory poisoning. The agent reads supplier emails, invoices, tool results and its own memory — all text an attacker can influence — and proposes payments. Before each decision, Tell reads the model's hidden state, scores it with a linear probe, and routes risky decisions to a safety-tuned LoRA of the same model. A deterministic validator and action gate hold the only authority to execute a payment.

Everything runs locally on one HP ZGX Nano (NVIDIA GB10). All money is simulated.

## How it works

```
invoice PDF ─► intake (Poppler text / Tesseract OCR, CPU) ─► extracted fields
                                                               │
                         ┌─────────────────────────────────────┘
                         ▼
   Qwen3-8B forward pass (base weights) ── hidden state, layer 27 ──► Tell probe (logistic regression) ─► score
                         │
        score < 0.1708   │  0.1708 – 0.5135        │  ≥ 0.5135
        Agent 1 (base)   │  Agent S (LoRA), no alarm│  Agent S + alarm latched
                         ▼
               proposed action (typed tool call)
                         ▼
   payment validator (always on) ─► action gate (blocks payments while alarm unresolved) ─► simulated SQLite ledger
```

| Component | What it is | Code |
|---|---|---|
| Agent 1 | Qwen3-8B, BF16, pinned revision `b968826d`. Can only *propose* actions | `src/tell/agent/` |
| Tell probe | Standardised layer-27 hidden state → logistic regression. 1–3 ms per score | `src/tell/detector/`, `results/probe_training/enterprise_v1/frozen_probe/` |
| Agent S | Same base model + PEFT LoRA (7.67 M trainable params, 0.09 %). Proposes verification, quarantine or evidence reports | `src/tell/safety/adapter_runtime.py`, `results/lora_training/agent_s_v1/frozen_adapter/` |
| Validator + gate | Deterministic checks against the trusted vendor master / ERP; the gate refuses `pay_invoice` while an alarm is latched | `src/tell/safety/` |
| Ledger, memory | SQLite simulation with provenance and quarantine | `src/tell/payment/`, `src/tell/memory/` |
| Prototype UI | Invoice intake, live agent trace, review queue, ledger | `demo/ui_ocr/` |

Probe and adapter files are hash-verified against their `FROZEN.sha256` before loading and are opened read-only.

## Local / hybrid inference

**All model inference is local.** No hosted LLM API is called at any point — Tell needs the model's hidden activations, which hosted and OpenAI-compatible endpoints (vLLM etc.) do not expose.

- **In-process model.** Qwen3-8B loads through Hugging Face Transformers with `local_files_only=True`, BF16, SDPA attention, on `cuda:0` (`src/tell/agent/local_model.py`). It refuses to run from a repo id, so it never downloads at runtime.
- **One model, two agents.** Agent 1 and Agent S share one copy of the weights; the LoRA is attached once and toggled per decision (`AdapterAwareRuntime`). Activation capture always runs with the adapter disabled.
- **Hybrid CPU/GPU split.** The prototype is two processes on the same machine:
  - *UI server* (`demo/ui_ocr/server.py`): Python stdlib only, CPU. Serves the UI, validates uploads, extracts text with Poppler and OCR with Tesseract.
  - *Live worker* (`demo/ui_ocr/live/worker.py`): the GPU process. Holds the model, probe and adapter; runs the agent loop, validator, gate and ledger. Jobs are handed over through a local SQLite queue; one GPU job at a time.
- **Replay mode.** Without a GPU, the UI serves stored traces of real agent runs (`demo/ui_ocr/data/traces_v1.json`) so the full interface works on any Linux machine.
- **Data stays on-device.** Invoices, vendor records, memory, ledger and audit events never leave the machine. An optional Cloudflare tunnel exposes only a passcode-protected UI serving a sanitised trace bundle.

Measured on the ZGX Nano: ~17 GB GPU memory for inference; cold model load ~100 s once per process; capture pass 1.5–2.2 s per scored decision; full clean-payment run ~31–33 s.

## Requirements

| Mode | Needs |
|---|---|
| Replay (UI only) | Linux, Python 3, `poppler-utils`; `tesseract-ocr` for scanned PDFs |
| Live | The above + NVIDIA GPU with ≥ 24 GB memory, CUDA driver for CUDA 13, Python 3.11/3.12, ~20 GB disk for the model |

Developed on the HP ZGX Nano (GB10, aarch64, Ubuntu 24.04, Python 3.12, torch 2.9.0+cu130).

## Setup

```bash
git clone https://github.com/aakashvardhan/accounts-payable-tell-agent.git
cd accounts-payable-tell-agent
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng

./setup.sh            # live: .venv, CUDA torch, pinned deps, Qwen3-8B download (~16 GB), artifact hash checks
./setup.sh --replay   # UI only: checks system tools, installs nothing
```

`setup.sh` options:

| Variable / flag | Effect |
|---|---|
| `--skip-model` | Skip the Qwen3-8B download |
| `TELL_MODEL_SNAPSHOT=/path` | Use an existing local snapshot of `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| `HF_HOME`, `HF_HUB_CACHE` | Where the model is downloaded / looked up |
| `TORCH_INDEX_URL` | Torch wheel index (default CUDA 13.0: `https://download.pytorch.org/whl/cu130`) |
| `PYTHON=python3.11` | Interpreter for `.venv` |

## Run

```bash
./run.sh live      # UI + GPU worker; upload invoices and watch the agent decide
./run.sh replay    # UI only, stored runs
./run.sh status
./run.sh stop
./run.sh test      # CPU test suites
```

Open `http://127.0.0.1:8084`. `HOST=0.0.0.0` exposes the UI to your LAN (no authentication); `PORT=` changes the port. Logs: `demo/ui_ocr/run/`.

In the UI: **Add invoices** (or **Check for new invoices** to load the samples in `runtime/inbox/`) → each invoice is extracted, then the live worker runs the agent and Tell. Open a case to see the Tell score, the route taken, the proposed action, the validator/gate decision and the ledger result. Held payments appear under **Human review**.

## Tests

`./run.sh test` runs:

- `pytest tests/` — agent, tools, probe capture, gate, ledger, routing. 719 pass; 38 need the locally built DocILE corpora and training outputs, which are not in the repository, and fail without them.
- `python3 -m unittest discover -s demo/ui_ocr/tests` — UI server, intake, OCR, auth, live runtime (207 tests; needs `python3-reportlab`).
- `node --test demo/ui_ocr/tests/` — UI logic (49 tests).

## Results

Numbers from `docs/TELL_METRICS_LEDGER.csv` (IDs in brackets).

| Question | Result |
|---|---|
| Detector, held-out immediate attacks | AUROC **0.841** (400 records, 50 documents) [PRB-09] |
| Detector, delayed memory poisoning | AUROC **0.609** — weak transfer [PRB-15] |
| Correct next action (1,400 cases) | Agent S **1,135** vs base **836**; premature payments 11 vs 51. Agent S regressed on tool-result forgery and lexical invoice injection [ARM-08, ARM-04, ARM-06] |
| Correct clean payments (76) | Tell-routed **76**, always-on Agent S 70, base 74 [ARM-03] |
| Unauthorized payments (700 attacked cases) | Base 1, Agent S 0, Tell-routed 0 — too sparse to claim theft reduction [ARM-01, ARM-02] |
| Validator + gate containment | Unit tests and 5 live development runs; not evaluated on held-out data [LIV-03, LIV-10] |

## Repository layout

```
src/tell/        agent loop and tools, detector (capture, probe), safety (LoRA runtime, gate, validator),
                 payment ledger, memory store, routing, registry, evaluation
scripts/         corpus builders, activation capture, probe/LoRA training, experiment runners, integrity hashing
configs/         frozen experiment, probe, LoRA and routing configurations
results/         frozen probe, frozen Agent-S adapter, operational threshold (only these are committed)
demo/ui_ocr/     the prototype (UI server, intake/OCR, live worker)
demo/ui, demo/ui_intake, demo/demo_a/   earlier UI iterations
docs/            metrics ledger, evidence manifest, presentation fact sheet
tests/           pytest suite
```

## Reproducing training

The probe and LoRA are trained on corpora derived from the [DocILE](https://docile.rossum.ai/) invoice dataset, which must be obtained under its own licence and placed in `data/docile/`. The pipeline is in `scripts/` (`build_enterprise_corpus_v2_2.py` → `capture_probe_corpus_activations_v1_1.py` → `probe_enterprise_v1/` → `agent_s_v1/train.py`). These research scripts use absolute paths from the development machine and need editing before running elsewhere. The prototype does not need them: the frozen artifacts are committed.

## Limitations

- Research prototype; not production-ready, not connected to any payment rail.
- Delayed-memory detection is weak (AUROC 0.609).
- Thresholds 0.1708 / 0.5135 are routing thresholds, not scores.
