#!/bin/sh
# Stop ONLY the exact recorded demo server process, after validating its command line. Logs are kept.
D="$(cd "$(dirname "$0")" && pwd)"; PIDF="$D/run/server.pid"
[ -f "$PIDF" ] || { echo "not running (no pid file)"; exit 0; }
P="$(cat "$PIDF")"
if [ -z "$P" ] || ! kill -0 "$P" 2>/dev/null; then echo "process $P not alive; clearing pid file"; rm -f "$PIDF" "$D/run/server.json"; exit 0; fi
if ! tr '\0' ' ' < /proc/$P/cmdline | grep -q "demo/ui/server.py"; then
  echo "refusing: pid $P is not the Tell demo server ($(tr '\0' ' ' < /proc/$P/cmdline))"; exit 1
fi
kill -TERM "$P"; i=0; while kill -0 "$P" 2>/dev/null && [ $i -lt 50 ]; do i=$((i+1)); sleep 0.1; done
kill -0 "$P" 2>/dev/null && { echo "pid $P still alive after TERM; not escalating"; exit 1; }
rm -f "$PIDF" "$D/run/server.json"; echo "stopped pid $P (log preserved: $D/run/server.log)"
