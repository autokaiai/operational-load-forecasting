#!/usr/bin/env bash
# Verify that ./checkpoint.pt is the graded artifact, byte for byte.
#
# The checkpoint ships with this repository, so nothing needs downloading. This script exists
# because a checkpoint that does not hash to the value below is NOT the model any number in this
# repository describes — and predict.py's ladder would still emit a well-formed CSV from a wrong
# one, which is exactly why the hash is checked rather than trusted.
#
#   ./scripts/verify_checkpoint.sh                 # verify ./checkpoint.pt
#   ./scripts/verify_checkpoint.sh path/to/ckpt    # verify some other copy
set -euo pipefail

TARGET="${1:-checkpoint.pt}"
SHA256="13e2f2d87dafd96ae3a7a14aaa5dc6ebc5e9a3fd7b0d1141bba7ce8328e0d295"

if [[ ! -f "$TARGET" ]]; then
  echo "[verify] no such file: $TARGET" >&2
  exit 2
fi

echo "[verify] $TARGET"
echo "$SHA256  $TARGET" | sha256sum -c -

echo "[verify] blend weights:"
unzip -p "$TARGET" bundle.json | python3 -c 'import json,sys; print(" ", json.load(sys.stdin)["blend"])'
