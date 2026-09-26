#!/usr/bin/env bash
# Run the Tell prototype (demo/ui_ocr).
#
#   ./run.sh replay   UI + PDF intake/OCR on CPU; stored agent runs are replayed, no model loaded
#   ./run.sh live     UI + live worker: Qwen3-8B + Tell probe + Agent-S LoRA process uploaded invoices on the GPU
#   ./run.sh stop     stop the UI server and the live worker
#   ./run.sh status   show what is running
#   ./run.sh test     run the CPU test suites
#
# Environment overrides: HOST (default 127.0.0.1; 0.0.0.0 exposes the UI to your LAN without auth), PORT (default 8084).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
UI="$ROOT/demo/ui_ocr"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8084}"
VPY="$ROOT/.venv/bin/python"

# runtime/ is local state (gitignored); seed its demo inbox with the committed sample invoices on first run
seed_inbox() { mkdir -p "$UI/runtime/inbox"; ls "$UI/runtime/inbox"/*.pdf >/dev/null 2>&1 || cp "$UI/runtime_public/inbox/"*.pdf "$UI/runtime/inbox/"; }

start_ui() { seed_inbox; "$UI/start_demo.sh" --host "$HOST" --port "$PORT" "$@"; }

case "${1:-}" in
  replay)
    start_ui
    echo "Open http://127.0.0.1:$PORT  (stop: ./run.sh stop)"
    ;;
  live)
    [ -x "$VPY" ] || { echo "no .venv: run ./setup.sh first"; exit 1; }
    "$UI/start_live_worker.sh" "$UI/runtime"
    start_ui --live-agent
    echo "Open http://127.0.0.1:$PORT  (stop: ./run.sh stop)"
    echo "The worker loads Qwen3-8B once (~100 s cold). Follow it with: tail -f $UI/run/live_worker.log"
    ;;
  stop)
    "$UI/stop_demo.sh" || true
    "$UI/stop_live_worker.sh" || true
    ;;
  status)
    for f in server live_worker; do
      p="$UI/run/$f.pid"
      if [ -f "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null; then echo "$f: running (pid $(cat "$p"))"; else echo "$f: stopped"; fi
    done
    [ -f "$UI/run/server.json" ] && grep -E '"(port|mode)"' "$UI/run/server.json" || true
    ;;
  test)
    PY="$VPY"; [ -x "$PY" ] || PY=python3
    CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q "$ROOT/tests" || true   # data-dependent tests fail without the local DocILE corpus
    (cd "$ROOT" && python3 -m unittest discover -s demo/ui_ocr/tests)   # system python3 + reportlab, like the UI server
    command -v node >/dev/null && node --test "$UI/tests/" || echo "node not found: skipped UI JS tests"
    ;;
  *)
    sed -n 2,11p "$0"; exit 2 ;;
esac
