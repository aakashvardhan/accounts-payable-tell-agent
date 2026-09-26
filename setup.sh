#!/usr/bin/env bash
# Set up Tell on a new Linux machine.
#
#   ./setup.sh              full setup: venv, CUDA torch, dependencies, Qwen3-8B download, artifact checks
#   ./setup.sh --replay     UI only: checks system tools; no venv, no GPU, no model
#   ./setup.sh --skip-model full setup without downloading Qwen3-8B (use an existing snapshot via TELL_MODEL_SNAPSHOT)
#
# Environment overrides:
#   PYTHON=python3.12                     interpreter used to create .venv (3.11 or 3.12)
#   TORCH_INDEX_URL=<url>                 torch wheel index (default: CUDA 13.0 wheels)
#   TELL_MODEL_SNAPSHOT=/path/to/snapshot use an existing local Qwen3-8B snapshot
#   HF_HOME / HF_HUB_CACHE                where the model is downloaded
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_REPO="Qwen/Qwen3-8B"
MODEL_REV="b968826d9c46dd6066d109eabc6255188de91218"
TORCH_VERSION="2.9.0"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

MODE=full; SKIP_MODEL=0
for a in "$@"; do
  case "$a" in
    --replay) MODE=replay ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown option: $a"; exit 2 ;;
  esac
done

ok()   { echo "  [ok]   $*"; }
warn() { echo "  [warn] $*"; }
die()  { echo "  [fail] $*"; exit 1; }

echo "== system"
[ "$(uname -s)" = "Linux" ] || die "Linux required (scripts use /proc and setsid)"
command -v python3 >/dev/null || die "python3 not found"
ok "python3 $(python3 -c 'import sys;print(sys.version.split()[0])') (UI server, stdlib only)"

MISSING=()
for t in pdftotext pdfinfo pdftoppm; do command -v "$t" >/dev/null || MISSING+=(poppler-utils); done
if command -v tesseract >/dev/null || [ -x "$ROOT/demo/ui_ocr/tools/tesseract/root/usr/bin/tesseract" ]; then
  ok "tesseract"
else
  MISSING+=(tesseract-ocr tesseract-ocr-eng)
fi
if [ ${#MISSING[@]} -gt 0 ]; then
  PKGS="$(printf '%s\n' "${MISSING[@]}" | sort -u | tr '\n' ' ')"
  warn "missing: $PKGS-> sudo apt-get install -y $PKGS"
  warn "without Poppler the UI cannot read uploaded PDFs; without Tesseract scanned PDFs are not OCR'd"
else
  ok "poppler-utils"
fi
python3 -c 'import reportlab' 2>/dev/null && ok "reportlab (UI tests, sample PDFs)" || warn "reportlab missing for system python3 (only UI tests / sample PDFs): sudo apt-get install -y python3-reportlab"
command -v node >/dev/null && ok "node (optional, UI unit tests)" || warn "node not found (optional, only for UI unit tests)"

if [ "$MODE" = replay ]; then
  echo
  echo "Replay setup complete. Start with: ./run.sh replay"
  exit 0
fi

echo "== gpu"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: live mode needs an NVIDIA GPU (use ./setup.sh --replay for the UI only)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  [ok]   /'

echo "== python environment"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3; do
    command -v "$c" >/dev/null && "$c" -c 'import sys;exit(0 if (3,11)<=sys.version_info[:2]<=(3,12) else 1)' && { PY="$c"; break; }
  done
fi
[ -n "$PY" ] || die "Python 3.11 or 3.12 required (set PYTHON=...)"
[ -x "$ROOT/.venv/bin/python" ] || "$PY" -m venv "$ROOT/.venv"
VPY="$ROOT/.venv/bin/python"
ok ".venv ($("$VPY" -V))"
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
"$VPY" -m pip install -q -r "$ROOT/requirements.txt"
"$VPY" -c 'import torch; assert torch.cuda.is_available(), "torch cannot see a CUDA device"; print("  [ok]   torch", torch.__version__, "cuda", torch.version.cuda, "->", torch.cuda.get_device_name(0))'

echo "== model ($MODEL_REPO @ ${MODEL_REV:0:12})"
SNAP="$("$VPY" -c 'import sys; sys.path.insert(0, "'"$ROOT"'/src"); from tell.agent.local_model import PINNED_SNAPSHOT_PATH as p; print(p)' 2>/dev/null || true)"
if [ -n "$SNAP" ] && [ -f "$SNAP/config.json" ]; then
  ok "snapshot present: $SNAP"
elif [ "$SKIP_MODEL" = 1 ]; then
  warn "no snapshot at ${SNAP:-<unknown>}; set TELL_MODEL_SNAPSHOT before ./run.sh live"
else
  echo "  downloading ~16 GB (BF16 weights)..."
  "$ROOT/.venv/bin/hf" download "$MODEL_REPO" --revision "$MODEL_REV" \
    --include "*.json" "*.safetensors" "*.txt" >/dev/null
  SNAP="$("$VPY" -c 'import sys; sys.path.insert(0, "'"$ROOT"'/src"); from tell.agent.local_model import PINNED_SNAPSHOT_PATH as p; print(p)')"
  [ -f "$SNAP/config.json" ] || die "download finished but no snapshot at $SNAP (set TELL_MODEL_SNAPSHOT)"
  ok "snapshot: $SNAP"
fi

echo "== frozen artifacts"
for d in results/lora_training/agent_s_v1/frozen_adapter results/probe_training/enterprise_v1/frozen_probe; do
  (cd "$ROOT/$d" && sha256sum --quiet -c FROZEN.sha256) || die "hash mismatch in $d"
  ok "$d"
done
[ -f "$ROOT/results/routing_design/operational_threshold_v1/operational_threshold_v1.json" ] || die "operational threshold artifact missing"
ok "results/routing_design/operational_threshold_v1/operational_threshold_v1.json"

echo
echo "Setup complete. Start with: ./run.sh live   (or ./run.sh replay for the UI without the model)"
