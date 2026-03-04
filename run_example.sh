#!/usr/bin/env bash
# run_example.sh  –  build, run, and plot a seispie-wg example
#
# Usage:
#   ./run_example.sh          (interactive menu)
#   ./run_example.sh 1        (run example 1 without prompting)
#
# Requires: Rust toolchain, Python 3 with numpy / matplotlib

set -euo pipefail

WG_ROOT="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$WG_ROOT/examples"
BINARY="$WG_ROOT/target/release/seispie-wg"
PLOT_SCRIPT="$WG_ROOT/scripts/plot_results.py"

# ── Collect available examples ────────────────────────────────────────────
mapfile -t DIRS < <(find "$EXAMPLES_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
NDIRS=${#DIRS[@]}

if [ "$NDIRS" -eq 0 ]; then
  echo "No examples found in $EXAMPLES_DIR" >&2
  exit 1
fi

# ── Select example ────────────────────────────────────────────────────────
re='^[0-9]+$'
if [[ "${1:-}" =~ $re ]] && [ "$1" -ge 1 ] && [ "$1" -le "$NDIRS" ]; then
  IE=$1
else
  echo "Select an example to run (1-$NDIRS):"
  for ((i=0; i<NDIRS; i++)); do
    echo "  $((i+1))) $(basename "${DIRS[$i]}")"
  done
  while true; do
    read -rp "> " IE
    if [[ "$IE" =~ $re ]] && [ "$IE" -ge 1 ] && [ "$IE" -le "$NDIRS" ]; then
      break
    fi
    echo "Please enter a number between 1 and $NDIRS."
  done
fi

EXAMPLE_DIR="${DIRS[$((IE-1))]}"
EXAMPLE_NAME="$(basename "$EXAMPLE_DIR")"
echo ""
echo "══════════════════════════════════════════════"
echo "  Example : $EXAMPLE_NAME"
echo "  Dir     : $EXAMPLE_DIR"
echo "══════════════════════════════════════════════"

# ── Build (release) ───────────────────────────────────────────────────────
echo ""
echo "▶ Building seispie-wg (release)…"
(cd "$WG_ROOT" && cargo build --release 2>&1)
echo "  ✓ binary: $BINARY"

# ── Generate model (if script present) ───────────────────────────────────
if [ -f "$EXAMPLE_DIR/generate_model.py" ]; then
  echo ""
  echo "▶ Generating model…"
  python "$EXAMPLE_DIR/generate_model.py"
fi

# ── Clean previous output ─────────────────────────────────────────────────
echo ""
echo "▶ Clearing previous output…"
rm -rf "$EXAMPLE_DIR/output"

# ── Step 1: prepare observed traces if a separate forward config exists ───
# Examples with config_true.ini run a forward pass first (model_true → obs_traces).
if [ -f "$EXAMPLE_DIR/config_true.ini" ]; then
  echo ""
  echo "▶ Generating observed traces (forward with model_true)…"
  rm -rf "$EXAMPLE_DIR/obs_traces" "$EXAMPLE_DIR/obs_trash"
  START=$(date +%s%3N)
  RUST_LOG=seispie_wg=info "$BINARY" "$EXAMPLE_DIR/config_true.ini"
  END=$(date +%s%3N)
  echo "  ✓ done in $(( END - START )) ms"
fi

# ── Run main solver ───────────────────────────────────────────────────────
echo ""
echo "▶ Running simulation ($(grep -m1 'mode' "$EXAMPLE_DIR/config.ini" | tr -d ' '))…"
START=$(date +%s%3N)
RUST_LOG=seispie_wg=info "$BINARY" "$EXAMPLE_DIR/config.ini"
END=$(date +%s%3N)
ELAPSED=$(( END - START ))
echo "  ✓ done in ${ELAPSED} ms"

# ── Plot results ──────────────────────────────────────────────────────────
echo ""
echo "▶ Plotting results…"
python "$PLOT_SCRIPT" "$EXAMPLE_DIR"
echo "  ✓ PNG files written alongside output data"

echo ""
echo "Done.  Output:"
find "$EXAMPLE_DIR/output" -name "*.png" | sort | sed 's/^/  /'
# Also report model PNGs written by plot_results.py for adjoint mode
find "$EXAMPLE_DIR" -maxdepth 2 \( -path "*/model_init/*.png" -o -path "*/model_true/*.png" \) \
  | sort | sed 's/^/  /'
