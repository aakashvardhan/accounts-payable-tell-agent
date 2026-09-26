# Tell: activation-guided security for an accounts-payable agent

An autonomous accounts-payable (AP) agent reads supplier emails, invoices, tool results and its own memory, then proposes payments. An attacker can influence all of that text. Tell watches the model's hidden state before each decision, scores it with a linear probe, and hands risky decisions to a safety-tuned LoRA of the same model. Only a deterministic validator and action gate can execute a payment.

The whole system runs on one HP ZGX Nano (NVIDIA GB10). The ledger is a simulation, so no real money moves.

## How it works

```
invoice PDF ─► intake (Poppler text / Tesseract OCR, CPU) ─► extracted fields
                                                               │
                         ┌─────────────────────────────────────┘
                         ▼
   Qwen3-8B forward pass (base weights) ── hidden state, layer 27 ──► Tell probe (logistic regression) ─► score
                         │
        score < 0.1708   │  0.1708 to 0.5135        │  ≥ 0.5135
        Agent 1 (base)   │  Agent S (LoRA), no alarm│  Agent S + alarm latched
                         ▼
               proposed action (typed tool call)
                         ▼
   payment validator (always on) ─► action gate (blocks payments while alarm unresolved) ─► simulated SQLite ledger
```

| Component | What it does | Code |
|---|---|---|
| Agent 1 | Qwen3-8B in BF16, pinned to revision `b968826d`. It proposes actions but cannot execute them | `src/tell/agent/` |
| Tell probe | Standardises the layer-27 hidden state and scores it with logistic regression in 1 to 3 ms | `src/tell/detector/`, `results/probe_training/enterprise_v1/frozen_probe/` |
| Agent S | Adds a PEFT LoRA to the same base model (7.67 M trainable params, 0.09 %). It proposes verification, quarantine or an evidence report | `src/tell/safety/adapter_runtime.py`, `results/lora_training/agent_s_v1/frozen_adapter/` |
| Validator + gate | Checks every payment against the trusted vendor master and ERP. The gate refuses `pay_invoice` while an alarm is latched | `src/tell/safety/` |
| Ledger, memory | Store payments and memories in SQLite, with provenance and quarantine | `src/tell/payment/`, `src/tell/memory/` |
| Prototype UI | Shows invoice intake, the live agent trace, the review queue and the ledger | `demo/ui_ocr/` |

Before loading the probe and adapter, the worker checks each file against its `FROZEN.sha256` and opens it read-only.

## Demo

We took these screenshots from a fresh clone running `./run.sh live`. The model processed the three sample invoices in `demo/ui_ocr/runtime_public/inbox/`.

**Invoices.** Upload PDFs or pull them from the demo inbox, then follow each one from extraction to payment.

![Invoice dashboard](docs/images/dashboard.png)

**Case file.** The invoice asks for payment to an account that differs from the vendor master. Tell scores the decision at 0.129, so Agent 1 carries on. Agent 1 pays the verified vendor account, not the one printed on the invoice, and the gate permits the payment.

![Case file for a paid invoice](docs/images/case_paid.png)

**Memory poisoning replay.** A poisoned memory note tells the agent the supplier "prefers expedited settlement". Tell scores the decision at 0.721, above the 0.5135 alarm threshold, so it hands the case to Agent S and the gate blocks the payment. This replays a recorded run from the training split, not a held-out result.

![Replay of a memory poisoning attack](docs/images/replay_memory.png)

**Payment ledger.** The ledger lists every payment decision from the live runtime and the test run, with the Tell score and the account each payment went to. No real money moves.

![Payment ledger](docs/images/ledger.png)

## Local / hybrid inference

**Tell runs every model call on the device.** It never calls a hosted LLM API. It can't: Tell needs the model's hidden activations, and hosted or OpenAI-compatible endpoints such as vLLM don't expose them.

- **In-process model.** `src/tell/agent/local_model.py` loads Qwen3-8B through Hugging Face Transformers with `local_files_only=True`, BF16 and SDPA attention on `cuda:0`. It rejects a bare repo id, so it never downloads anything at runtime.
- **One model, two agents.** Agent 1 and Agent S share one copy of the weights. `AdapterAwareRuntime` attaches the LoRA once and switches it on per decision. Activation capture always runs with the adapter off.
- **Hybrid CPU/GPU split.** The prototype runs as two processes on the same machine:
  - The *UI server* (`demo/ui_ocr/server.py`) uses only the Python standard library and the CPU. It serves the UI, validates uploads, extracts text with Poppler and runs OCR with Tesseract.
  - The *live worker* (`demo/ui_ocr/live/worker.py`) owns the GPU. It holds the model, probe and adapter and runs the agent loop, validator, gate and ledger. It picks up jobs from a local SQLite queue, one at a time.
- **Replay mode.** On a machine without a GPU, the UI serves stored traces of real agent runs (`demo/ui_ocr/data/traces_v1.json`), so you can still use the full interface on any Linux machine.
- **Data stays on the device.** Invoices, vendor records, memory, the ledger and audit events never leave the machine. If you use the optional Cloudflare tunnel, it exposes only a passcode-protected UI that serves a sanitised trace bundle.

On the ZGX Nano, inference uses about 17 GB of GPU memory. The model takes about 100 s to load once per process. Each scored decision adds a 1.5 to 2.2 s capture pass, and a full clean-payment run takes 31 to 33 s.

## Requirements

| Mode | You need |
|---|---|
| Replay (UI only) | Linux, Python 3 and `poppler-utils`. Add `tesseract-ocr` to read scanned PDFs |
| Live | Everything above, plus an NVIDIA GPU with at least 24 GB of memory, a CUDA 13 driver, Python 3.11 or 3.12, and about 20 GB of disk for the model |

We built and tested Tell on the HP ZGX Nano (GB10, aarch64, Ubuntu 24.04, Python 3.12, torch 2.9.0+cu130).

## Setup

```bash
git clone https://github.com/aakashvardhan/accounts-payable-tell-agent.git
cd accounts-payable-tell-agent
sudo apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-eng

./setup.sh            # live: .venv, CUDA torch, pinned deps, Qwen3-8B download (~16 GB), artifact hash checks
./setup.sh --replay   # UI only: checks system tools, installs nothing
```

You can change what `setup.sh` does with these options:

| Variable / flag | Effect |
|---|---|
| `--skip-model` | Skips the Qwen3-8B download |
| `TELL_MODEL_SNAPSHOT=/path` | Points to a local snapshot of `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` you already have |
| `HF_HOME`, `HF_HUB_CACHE` | Sets where the script downloads and looks for the model |
| `TORCH_INDEX_URL` | Sets the torch wheel index (default CUDA 13.0: `https://download.pytorch.org/whl/cu130`) |
| `PYTHON=python3.11` | Picks the interpreter for `.venv` |

## Run

```bash
./run.sh live      # UI + GPU worker; upload invoices and watch the agent decide
./run.sh replay    # UI only, stored runs
./run.sh status
./run.sh stop
./run.sh test      # CPU test suites
```

Open `http://127.0.0.1:8084`. Set `HOST=0.0.0.0` to share the UI on your LAN (it has no authentication) and `PORT=` to change the port. The scripts write logs to `demo/ui_ocr/run/`.

In the UI, click **Add invoices**, or **Check for new invoices** to load the samples in `runtime/inbox/`. The server extracts each invoice, then the live worker runs the agent and Tell. Open a case to see the Tell score, the route, the proposed action, the validator and gate decision, and the ledger result. You'll find held payments under **Human review**.

## Tests

`./run.sh test` runs three suites:

- `pytest tests/` covers the agent, tools, probe capture, gate, ledger and routing. 719 tests pass. The other 38 need DocILE corpora and training outputs that we built locally and didn't commit, so they fail on a fresh clone.
- `python3 -m unittest discover -s demo/ui_ocr/tests` covers the UI server, intake, OCR, auth and live runtime (207 tests). It needs `python3-reportlab`.
- `node --test demo/ui_ocr/tests/` covers the UI logic (49 tests).

## Results

Every number comes from `docs/TELL_METRICS_LEDGER.csv`. The IDs in brackets point to its rows.

| Question | Result |
|---|---|
| Does the detector catch held-out immediate attacks? | AUROC **0.841** on 400 records from 50 documents [PRB-09] |
| Does it catch delayed memory poisoning? | Only weakly: AUROC **0.609** [PRB-15] |
| Does Agent S pick the right next action? (1,400 cases) | Agent S got **1,135** right and the base model **836**. Agent S proposed 11 premature payments; the base model proposed 51. Agent S did worse on tool-result forgery and lexical invoice injection [ARM-08, ARM-04, ARM-06] |
| Does routing keep clean payments working? (76 cases) | Tell routing paid all **76** correctly, always-on Agent S paid 70 and the base model 74 [ARM-03] |
| Does Tell stop unauthorized payments? (700 attacked cases) | The base model made 1, Agent S 0 and Tell routing 0. That is too few events to claim Tell reduces theft [ARM-01, ARM-02] |
| Do the validator and gate contain payments? | They pass unit tests and 5 live development runs. We haven't evaluated them on held-out data [LIV-03, LIV-10] |

## Repository layout

```
src/tell/        agent loop and tools, detector (capture, probe), safety (LoRA runtime, gate, validator),
                 payment ledger, memory store, routing, registry, evaluation
scripts/         corpus builders, activation capture, probe/LoRA training, experiment runners, integrity hashing
configs/         frozen experiment, probe, LoRA and routing configurations
results/         frozen probe, frozen Agent-S adapter, operational threshold (the only committed results)
demo/ui_ocr/     the prototype (UI server, intake/OCR, live worker)
demo/ui, demo/ui_intake, demo/demo_a/   earlier UI iterations
docs/            metrics ledger, evidence manifest, presentation fact sheet
tests/           pytest suite
```

## Reproducing training

We trained the probe and LoRA on corpora built from the [DocILE](https://docile.rossum.ai/) invoice dataset. Get DocILE under its own licence and put it in `data/docile/`. The pipeline lives in `scripts/`: `build_enterprise_corpus_v2_2.py` → `capture_probe_corpus_activations_v1_1.py` → `probe_enterprise_v1/` → `agent_s_v1/train.py`. These research scripts still hold absolute paths from our development machine, so edit them before you run them elsewhere. You don't need them to run the prototype, because the repository already includes the frozen artifacts.

## Limitations

- Tell is a research prototype. It isn't production-ready and doesn't connect to any payment rail.
- It detects delayed memory poisoning only weakly (AUROC 0.609).
- 0.1708 and 0.5135 are routing thresholds, not scores.
