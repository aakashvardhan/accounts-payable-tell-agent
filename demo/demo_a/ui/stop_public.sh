#!/bin/sh
# Stop ONLY the public tunnel and the public (8082) server, after validating each recorded pid's command line.
D="$(cd "$(dirname "$0")" && pwd)/run"
stop(){ # name pidfile pattern
  [ -f "$2" ] || { echo "$1: no pid file"; return; }
  P="$(cat "$2")"; kill -0 "$P" 2>/dev/null || { echo "$1: pid $P not alive"; rm -f "$2"; return; }
  C="$(tr '\0' ' ' < /proc/$P/cmdline)"
  case "$C" in *"$3"*) kill -TERM "$P"; i=0; while kill -0 "$P" 2>/dev/null && [ $i -lt 50 ]; do i=$((i+1)); sleep 0.1; done
      kill -0 "$P" 2>/dev/null && echo "$1: pid $P still alive" || { rm -f "$2"; echo "$1: stopped pid $P"; } ;;
    *) echo "$1: refusing, pid $P is not the expected process ($C)";; esac; }
stop tunnel "$D/public_tunnel.pid" "cloudflared-2026.9.3 tunnel --no-autoupdate --url http://127.0.0.1:8082"
stop public-server "$D/public_server.pid" "server.py --host 127.0.0.1 --port 8082"
rm -f "$D/public_server_info.json"; echo "logs preserved in $D"
