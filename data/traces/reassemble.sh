#!/usr/bin/env bash
# Reassemble large trace tarballs from their split parts.
#
# GitHub blobs are capped at 100 MB so each archive was split with
#   split -b 90M -d --suffix-length=2 <file>.tar.gz <file>.tar.gz.part_
# into part_00, part_01, ... This script concatenates them back, verifies
# the sha256, and (optionally) extracts the directory.
#
# Currently handled archives:
#   - trn2_48_llama3_1_8b_instruct
#   - trn2_3_qwen1_5_moe
#
# Usage:
#   ./reassemble.sh                # reassemble all archives
#   ./reassemble.sh --extract      # also untar each into ./<name>/
#   ./reassemble.sh <name> [...]   # reassemble only the named archive(s)
#   ./reassemble.sh --extract <name> [...]
set -euo pipefail

cd "$(dirname "$0")"

# name : expected_sha256
declare -A ARCHIVES=(
    [trn2_48_llama3_1_8b_instruct]="14c580aa02fe0488374720cce96ae87081be462d9e11ce7804dce07b8870a90d"
    [trn2_3_qwen1_5_moe]="49480bf6410684ca6f897be055eb7127da77a318e78dde7be90f0b3fc6fe7f1e"
)

EXTRACT=0
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        --extract) EXTRACT=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) SELECTED+=("$arg") ;;
    esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    SELECTED=("${!ARCHIVES[@]}")
fi

reassemble_one() {
    local name="$1"
    local expected_sha="${ARCHIVES[$name]:-}"
    local out="${name}.tar.gz"
    local parts_dir="${name}.parts"

    if [[ -z "$expected_sha" ]]; then
        echo "[error] unknown archive: $name" >&2
        echo "        known: ${!ARCHIVES[*]}" >&2
        return 1
    fi
    if [[ ! -d "$parts_dir" ]]; then
        echo "[error] $parts_dir not found next to this script" >&2
        return 1
    fi

    echo "[reassemble] $name: concatenating parts -> $out"
    cat "$parts_dir"/"$name".tar.gz.part_?? > "$out"

    echo "[reassemble] $name: verifying sha256"
    local actual_sha
    actual_sha=$(sha256sum "$out" | awk '{print $1}')
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "[error] sha256 mismatch for $out" >&2
        echo "        expected: $expected_sha" >&2
        echo "        actual:   $actual_sha" >&2
        return 1
    fi
    echo "[ok] $name: sha256 matches: $actual_sha"

    if [[ "$EXTRACT" -eq 1 ]]; then
        echo "[reassemble] $name: extracting to ./$name/"
        tar -xzf "$out"
        echo "[done] $name: extracted."
    fi
}

rc=0
for name in "${SELECTED[@]}"; do
    if ! reassemble_one "$name"; then
        rc=1
    fi
done

if [[ "$EXTRACT" -ne 1 ]]; then
    echo "[done] re-run with --extract to also untar."
fi

exit "$rc"
