#!/bin/sh
# Stop ONLY the recorded live worker (validates its command line; finishes the current job first via SIGTERM).
D="$(cd "$(dirname "$0")" && pwd)"; PIDF="$D/run/live_worker.pid"
[ -f "$PIDF" ] || { echo "no live worker pid file"; exit 0; }
P="$(cat "$PIDF")"
kill -0 "$P" 2>/dev/null || { echo "process $P not alive; clearing pid file"; rm -f "$PIDF"; exit 0; }
tr '\0' ' ' < /proc/$P/cmdline | grep -q "live/worker.py" || { echo "refusing: pid $P is not the live worker"; exit 1; }
kill -TERM "$P"; i=0; while kill -0 "$P" 2>/dev/null && [ $i -lt 300 ]; do i=$((i+1)); sleep 0.2; done
kill -0 "$P" 2>/dev/null && { echo "pid $P still alive after 60s; not escalating"; exit 1; }
rm -f "$PIDF"; echo "stopped live worker pid $P (log preserved)"
