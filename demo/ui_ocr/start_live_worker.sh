#!/bin/sh
# Start the live runtime worker (real Qwen3-8B + frozen LoRA + frozen probe) in the background. Refuses a duplicate.
# usage: start_live_worker.sh <runtime-dir> [<runtime-dir> ...]   (extra args after -- are passed to the worker)
D="$(cd "$(dirname "$0")" && pwd)"; RUN="$D/run"; mkdir -p "$RUN"
PIDF="$RUN/live_worker.pid"; LOG="$RUN/live_worker.log"
if [ -f "$PIDF" ]; then
  P="$(cat "$PIDF")"
  if [ -n "$P" ] && kill -0 "$P" 2>/dev/null && tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "live/worker.py"; then
    echo "live worker already running (pid $P)"; exit 1
  fi
  rm -f "$PIDF"
fi
ARGS=""; for d in "$@"; do ARGS="$ARGS --runtime-dir $d"; done
[ -n "$ARGS" ] || { echo "usage: $0 <runtime-dir> [...]"; exit 2; }
echo "=== start $(date -Is) ===" >> "$LOG"
cd "$D" || exit 1
nohup setsid /home/hp5/tell/.venv/bin/python "$D/live/worker.py" $ARGS >> "$LOG" 2>&1 &
echo $! > "$PIDF"; echo "live worker starting (pid $(cat $PIDF)); log: $LOG"
