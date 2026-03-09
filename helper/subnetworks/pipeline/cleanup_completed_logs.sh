#!/bin/bash
set -euo pipefail

LOG_DIR="${1:-.}"
HOURS_OLD="${2:-4}"
MARKER="${3:-TIME_ENDED:}"

if [[ ! -d "${LOG_DIR}" ]]; then
  exit 0
fi

LOCK_DIR="${LOG_DIR}/.log_cleanup.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" >/dev/null 2>&1 || true' EXIT

while IFS= read -r -d '' file_path; do
  if grep -q "${MARKER}" "${file_path}" 2>/dev/null; then
    rm -f "${file_path}"
  fi
done < <(find "${LOG_DIR}" -type f -mmin "+$((HOURS_OLD * 60))" -print0)
