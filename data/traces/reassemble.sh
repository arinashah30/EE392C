#!/usr/bin/env bash
# Reassemble GitHub-100MB-friendly split tarballs.
#
# Each dataset lives next to this script as:
#   <name>.parts/
#     <name>.tar.gz.part_00
#     <name>.tar.gz.part_01
#     ...
#     <name>.tar.gz.sha256        # one line: expected sha256 of the full tar.gz
#
# Running this script concatenates the parts back into <name>.tar.gz at this
# directory, verifies the sha256, and (optionally) untars to ./<name>/.
#
# Usage:
#   ./reassemble.sh                                  # reassemble every dataset found here
#   ./reassemble.sh --extract                        # reassemble + untar every dataset
#   ./reassemble.sh trn2_48_qwen1_5_moe              # only one dataset
#   ./reassemble.sh trn2_48_qwen1_5_moe --extract
#   ./reassemble.sh trn2_48_llama3_1_8b_instruct trn2_48_qwen1_5_moe --extract
set -euo pipefail

cd "$(dirname "$0")"

EXTRACT=0
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        --extract|-x)    EXTRACT=1 ;;
        -h|--help)
            sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        --*)
            echo "[error] unknown flag: $arg" >&2
            exit 2
            ;;
        *) SELECTED+=("$arg") ;;
    esac
done

reassemble_one() {
    local name="$1"
    local parts_dir="${name}.parts"
    local out="${name}.tar.gz"
    local sha_file="${parts_dir}/${name}.tar.gz.sha256"

    if [[ ! -d "$parts_dir" ]]; then
        echo "[error] $parts_dir/ not found next to this script" >&2
        return 1
    fi
    if [[ ! -f "$sha_file" ]]; then
        echo "[error] $sha_file not found (expected sidecar sha256)" >&2
        return 1
    fi

    echo "==> $name"
    echo "    parts -> $out"
    # shellcheck disable=SC2086
    cat "$parts_dir"/${name}.tar.gz.part_?? > "$out"

    local expected actual
    expected="$(tr -d ' \t\r\n' < "$sha_file")"
    actual="$(sha256sum "$out" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
        echo "    [error] sha256 mismatch" >&2
        echo "      expected: $expected" >&2
        echo "      actual:   $actual" >&2
        return 1
    fi
    echo "    [ok] sha256: $actual"

    if [[ "$EXTRACT" == "1" ]]; then
        echo "    extracting -> ./$name/"
        tar -xzf "$out"
        echo "    [done] $name/"
    fi
}

if [[ ${#SELECTED[@]} -gt 0 ]]; then
    for name in "${SELECTED[@]}"; do
        reassemble_one "$name"
    done
else
    shopt -s nullglob
    found=0
    for d in *.parts/; do
        reassemble_one "${d%.parts/}"
        found=$((found + 1))
    done
    if [[ $found -eq 0 ]]; then
        echo "[error] no *.parts/ directories found in $(pwd)" >&2
        exit 1
    fi
fi

if [[ "$EXTRACT" != "1" ]]; then
    echo
    echo "[done] re-run with --extract to also untar."
fi
