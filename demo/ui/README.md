# Tell demo interface (v1, REPLAY mode)

Local-network web UI for a recorded guided demo. CPU-only, stdlib Python + static HTML/CSS/JS. No model, no CUDA,
no CDN/fonts/remote JS, no package installs. Port 8080 is reserved for the future model/API service.

## Run
    demo/ui/start_demo.sh     # background; pid -> run/server.pid, log -> run/server.log, refuses duplicates
    demo/ui/stop_demo.sh      # stops only the recorded pid after validating its command line; log preserved
Prefers `0.0.0.0:8081`; if occupied it takes the next free port (never 8080) and logs why. `GET /healthz` is the health check.
The Same-network URL is derived from the active route at start-up (not hardcoded). Local network only — not internet-public.

## Layout
- `trace_contract.py` — `tell.demo_trace/1.0` schema + validator (per-field provenance, zone check, audit hash chain).
- `build_traces.py`  — reads (read-only) the epoch-0 replay `records.jsonl`/`metrics.json` and `configs/routing/tell_three_zone_v1.json`,
  overlays a deterministic fixture, writes `data/traces_v1.json`. Re-run: `python3 demo/ui/build_traces.py`.
- `server.py`, `static/{index.html,styles.css,tell_core.js,app.js}` — the UI consumes only the contract.

## Provenance rule
Every structured field is tagged `real` (captured epoch-0 replay, TRAINING split, not held-out) or `fixture`
(`SIMULATED DEMO FIXTURE — NOT MEASURED MODEL PERFORMANCE`). Tell score/routing/alarm/actions/lookups/tokens/latency/hashes/thresholds are real;
supplier names, message text, memory entry, validator approval (clean story), gate decision, ledger, outcomes, queue status, fleet
counters, audit ids and evidence-report text are fixture. The captured validator outcome is kept alongside any fixture approval.

## Tests (CPU only, focused)
    python3 -m unittest discover -s demo/ui/tests -v
    node --test demo/ui/tests/
