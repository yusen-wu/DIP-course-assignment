#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE="facades"
URL="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/${FILE}.tar.gz"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
TAR_FILE="${DATASETS_DIR}/${FILE}.tar.gz"
TARGET_DIR="${DATASETS_DIR}/${FILE}"
TRAIN_DIR="${TARGET_DIR}/train"
VAL_DIR="${TARGET_DIR}/val"
TRAIN_LIST="${SCRIPT_DIR}/train_list.txt"
VAL_LIST="${SCRIPT_DIR}/val_list.txt"

mkdir -p "${DATASETS_DIR}"
echo "Downloading ${URL} ..."

if command -v wget >/dev/null 2>&1; then
    wget -N "${URL}" -O "${TAR_FILE}"
elif command -v curl >/dev/null 2>&1; then
    curl -L "${URL}" -o "${TAR_FILE}"
else
    echo "Error: neither wget nor curl is available." >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"
tar -zxvf "${TAR_FILE}" -C "${DATASETS_DIR}"
rm -f "${TAR_FILE}"

find "${TRAIN_DIR}" -type f -name "*.jpg" | sort > "${TRAIN_LIST}"
find "${VAL_DIR}" -type f -name "*.jpg" | sort > "${VAL_LIST}"

echo "Saved training file list to ${TRAIN_LIST}"
echo "Saved validation file list to ${VAL_LIST}"
