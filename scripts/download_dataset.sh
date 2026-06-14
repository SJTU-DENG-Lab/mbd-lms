#!/bin/bash
set -e

REPO="SJTU-DENG-Lab/MBD-LMs-MultiTF-Datasets"

# Change DATA_ROOT to your preferred download location, or set DOWNLOAD_DIR in env.
DATA_ROOT="${DATA_ROOT:-$HOME/data/.cache/huggingface/datasets}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT}/${REPO}}"
TARGET_LINK="dataset"

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: 'hf' CLI not found. Install it: pip install -U huggingface_hub" >&2
    exit 1
fi

echo "============================================"
echo " Dataset Download & Link"
echo "============================================"
echo " Repository:   ${REPO}"
echo " Download to:  ${DOWNLOAD_DIR}"
echo " Symlink:      $(pwd)/${TARGET_LINK}"
echo "============================================"

if [ -d "${DOWNLOAD_DIR}" ]; then
    echo "Already downloaded, skipping."
else
    echo "Downloading..."
    mkdir -p "$(dirname "${DOWNLOAD_DIR}")"
    hf download "${REPO}" --repo-type dataset --local-dir "${DOWNLOAD_DIR}"
    echo "Download complete."
fi

# Remove existing symlink or directory
if [ -L "${TARGET_LINK}" ] || [ -d "${TARGET_LINK}" ]; then
    rm -rf "${TARGET_LINK}"
fi

mkdir -p "${TARGET_LINK}"

# Symlink each jsonl file into dataset/
shopt -s nullglob
jsonl_files=("${DOWNLOAD_DIR}"/*.jsonl)
if [ ${#jsonl_files[@]} -eq 0 ]; then
    echo "Warning: no .jsonl files found in ${DOWNLOAD_DIR}" >&2
else
    for f in "${jsonl_files[@]}"; do
        ln -s "${f}" "${TARGET_LINK}/$(basename "${f}")"
    done
    echo "Linked ${#jsonl_files[@]} .jsonl files into ${TARGET_LINK}/"
fi

echo "Done."
