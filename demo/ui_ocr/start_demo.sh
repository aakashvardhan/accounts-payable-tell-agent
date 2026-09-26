#!/bin/sh
# Start the Tell demo UI in the background. Refuses to start a duplicate. CPU-only, stdlib python3.
D="$(cd "$(dirname "$0")" && pwd)"; RUN="$D/run"; mkdir -p "$RUN"
PIDF="$RUN/server.pid"; LOG="$RUN/server.log"
if [ -f "$PIDF" ]; then
  P="$(cat "$PIDF")"
  if [ -n "$P" ] && kill -0 "$P" 2>/dev/null && tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -q "demo/ui_ocr/server.py"; then
    echo "already running (pid $P); see $RUN/server.json. Use stop_demo.sh first."; exit 1
  fi
  echo "removing stale pid file"; rm -f "$PIDF"
fi
rm -f "$RUN/server.json"
cd "$D" || exit 1
echo "=== start $(date -Is) ===" >> "$LOG"
nohup setsid "${PYTHON3:-python3}" "$D/server.py" "$@" >> "$LOG" 2>&1 &
echo $! > "$PIDF"
i=0; while [ $i -lt 50 ]; do [ -f "$RUN/server.json" ] && break; i=$((i+1)); sleep 0.1; done
[ -f "$RUN/server.json" ] || { echo "server failed to start; see $LOG"; rm -f "$PIDF"; tail -5 "$LOG"; exit 1; }
PORT="$("${PYTHON3:-python3}" -c "import json;print(json.load(open('$RUN/server.json'))['port'])")"
curl -fsS "http://127.0.0.1:$PORT/healthz" && echo || { echo "health check failed; see $LOG"; exit 1; }
tail -n 6 "$LOG" | grep -E "listening|Nano-local|Same-network|unavailable|internet"
