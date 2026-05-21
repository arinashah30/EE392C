#!/usr/bin/env bash
# Reassemble trn2_48_llama3_1_8b_instruct.tar.gz from its split parts.
#
# GitHub blobs are capped at 100 MB so the 200 MB archive was split with
#   split -b 90M -d --suffix-length=2 <file> <prefix>
# into part_00, part_01, part_02. This script concatenates them back and
# (optionally) extracts the directory.
#
# Usage:
#   ./reassemble.sh              # produces trn2_48_llama3_1_8b_instruct.tar.gz
#   ./reassemble.sh --extract    # also untars to ./trn2_48_llama3_1_8b_instruct/
set -euo pipefail

cd "$(dirname "$0")"

OUT="trn2_48_llama3_1_8b_instruct.tar.gz"
PARTS_DIR="trn2_48_llama3_1_8b_instruct.parts"
EXPECTED_SHA="14c580aa02fe0488374720cce96ae87081be462d9e11ce7804dce07b8870a90d"

if [[ ! -d "$PARTS_DIR" ]]; then
    echo "[error] $PARTS_DIR not found next to this script" >&2
    exit 1
fi

echo "[reassemble] concatenating parts -> $OUT"
cat "$PARTS_DIR"/trn2_48_llama3_1_8b_instruct.tar.gz.part_?? > "$OUT"

echo "[reassemble] verifying sha256"
ACTUAL_SHA=$(sha256sum "$OUT" | awk '{print $1}')
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "[error] sha256 mismatch" >&2
    echo "        expected: $EXPECTED_SHA" >&2
    echo "        actual:   $ACTUAL_SHA" >&2
    exit 1
fi
echo "[ok] sha256 matches: $ACTUAL_SHA"

if [[ "${1:-}" == "--extract" ]]; then
    echo "[reassemble] extracting to ./trn2_48_llama3_1_8b_instruct/"
    tar -xzf "$OUT"
    echo "[done] extracted."
else
    echo "[done] re-run with --extract to also untar."
fi
