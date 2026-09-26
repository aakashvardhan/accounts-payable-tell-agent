# Tell

Tell is a locally deployed security runtime for an autonomous accounts-payable
(AP) agent. The agent reads supplier emails and invoices, consults a vendor
record and persistent memory, and submits payments to a simulated fiat
ledger. Untrusted business content — a compromised email, an altered
invoice, a poisoned tool result, or a stored memory — can attempt to redirect
a payment without a human in the loop.

Tell tests whether the model's internal activations reveal that it is being
manipulated before it makes an unsafe tool call, and uses that signal to
route the agent to a safer response path and to stop an unresolved risk from
reaching the payment executor.

This is an Edge AI systems project, not a chatbot or a rules engine. It
combines an instrumented local model, an activation probe, a safety-tuned
LoRA, a deterministic action gate, a realistic agentic AP workflow, and an
evaluation harness that attacks several context surfaces.

Full project specification: [`Tell_Project_Knowledge.md`](Tell_Project_Knowledge.md).

## Status

Repository foundation only. No business logic, agent behavior, probe,
LoRA, gate policy, or ledger implementation has been written yet. See
[`CLAUDE.md`](CLAUDE.md) for the constraints that govern how this project is
built out.

## Central research question

> If a probe is trained to recognize immediate prompt injection in email and
> tool results, will the same internal signal detect poison after it has
> been stored, retrieved later, and presented as the agent's own memory?

## Runtime architecture (planned)

1. The user assigns an AP task (e.g. "Pay this month's approved invoices").
2. The base agent calls a read tool (email, invoice, vendor record, memory).
3. The tool result enters the agent's context.
4. An instrumented forward pass captures a hidden-state vector before the
   next action is generated.
5. The Tell probe returns an alarm score from that vector.
6. Below threshold: the base model generates the next action normally.
7. Above threshold: the same model runs with the safety LoRA enabled and
   proposes a quarantine, verification, or replanning action instead.
8. A deterministic action gate evaluates the proposed tool call against the
   current alarm state. Read-only and recovery actions may proceed; a
   payment (`pay_invoice`) is blocked while the alarm is unresolved.
9. The executor runs the permitted tool and records the result in the audit
   log.

| Component | Responsibility |
| --- | --- |
| Tell probe | Estimate whether the model is being steered by untrusted content |
| Base model | Complete ordinary AP work (Qwen3-8B instruct) |
| Safety LoRA | Generate safer actions when Tell fires (PEFT adapter) |
| Action gate | Deterministically prevent unresolved risk from executing |

## Local implementation stack

- Python 3.11
- PyTorch and Hugging Face Transformers for the instrumented forward pass
  (hidden-state hooks require an in-process runtime, not an opaque
  completion endpoint)
- PEFT for LoRA training and adapter switching
- scikit-learn for the first linear probe
- Pydantic for typed tool arguments and validation
- SQLite for vendor data, memory, ledger state, and audit events
- FastAPI for the local service API
- A lightweight local web interface (Streamlit or a simple React client) for
  the demo
- pytest for tool, gate, ledger, and scenario tests

All core inference, hidden-state capture, probe inference, safety-LoRA
routing, and agent execution run locally on the assigned HP ZGX Nano
(NVIDIA GB10 Grace Blackwell). See [`CLAUDE.md`](CLAUDE.md) for the exact
boundary of what may and may not leave the device.

## Repository layout

```text
tell/
  README.md
  CLAUDE.md
  pyproject.toml
  .gitignore
  Tell_Project_Knowledge.md
  configs/                  # run/experiment configuration (empty scaffold)
  data/
    docile_samples/         # local DocILE invoice samples (not committed)
    scenarios/               # generated clean/attack scenario pairs (not committed)
  src/tell/
    agent/                   # agent loop, prompts, typed tool schemas
    detector/                # hidden-state capture, probe, calibration
    safety/                  # LoRA adapter routing, deterministic action gate
    payment/                 # local simulated ledger
    memory/                  # persistent memory store and provenance
    evaluation/              # scenario runner and metrics
    api/                     # local FastAPI service
  tests/                     # pytest suite (empty scaffold)
  scripts/                   # data prep, activation collection, training, eval
  demo/                      # live-demo assets and UI (empty scaffold)
  results/                   # benchmark outputs (empty scaffold, gitignored)
```

## Setup

Not yet available. No dependencies have been installed, no datasets or
models have been downloaded, and no packages have been built. This will be
filled in once the clean AP workflow scaffold is implemented and reviewed.

## Deadline

Friday, September 25 at 8 PM.
