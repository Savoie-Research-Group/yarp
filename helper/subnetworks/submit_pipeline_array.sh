#!/bin/bash
# Submit UGE/qsub array jobs for run_pipeline.sh using manifest size.
set -euo pipefail

if [[ -n "${SGE_O_WORKDIR:-}" ]]; then
  ROOT_DIR="${SGE_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PIPELINE_CFG="${PIPELINE_CFG:-${ROOT_DIR}/pipeline/configs/pipeline_config.yaml}"
NETWORK_DIR="${NETWORK_DIR:-${ROOT_DIR}/networks}"
NETWORK_GLOB="${NETWORK_GLOB:-*.pkl}"
MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/networks/manifest.txt}"
AUTO_BUILD_MANIFEST="${AUTO_BUILD_MANIFEST:-1}"
QSUB_BIN="${QSUB_BIN:-qsub}"
ARRAY_MAX_TASKS="${ARRAY_MAX_TASKS:-10}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-1}"

if [[ "${AUTO_BUILD_MANIFEST}" == "1" ]]; then
  python - <<PY
from pathlib import Path
network_dir = Path(r"""${NETWORK_DIR}""").expanduser().resolve()
glob_pat = r"""${NETWORK_GLOB}"""
manifest_path = Path(r"""${MANIFEST_PATH}""").expanduser().resolve()
if not network_dir.exists():
    raise FileNotFoundError(f"Network directory does not exist: {network_dir}")
paths = sorted([p.resolve() for p in network_dir.glob(glob_pat) if p.is_file()])
if not paths:
    raise RuntimeError(f"No network files found in {network_dir} matching {glob_pat}")
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text("\\n".join(str(p) for p in paths) + "\\n")
print(f"Manifest written: {manifest_path} | entries={len(paths)}")
PY
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}"
  exit 1
fi

TOTAL_TASKS="$(grep -cve '^\s*$' "${MANIFEST_PATH}")"
if [[ "${TOTAL_TASKS}" -lt 1 ]]; then
  echo "Manifest has no entries: ${MANIFEST_PATH}"
  exit 1
fi

if [[ "${ARRAY_MAX_TASKS}" =~ ^[0-9]+$ ]] && [[ "${ARRAY_MAX_TASKS}" -gt 0 ]] && [[ "${ARRAY_MAX_TASKS}" -lt "${TOTAL_TASKS}" ]]; then
  TASKS_TO_RUN="${ARRAY_MAX_TASKS}"
else
  TASKS_TO_RUN="${TOTAL_TASKS}"
fi

echo "Submitting UGE array:"
echo "  pipeline_cfg=${PIPELINE_CFG}"
echo "  manifest=${MANIFEST_PATH}"
echo "  total_manifest_tasks=${TOTAL_TASKS}"
echo "  array_tasks_to_run=${TASKS_TO_RUN}"
echo "  task_concurrency=${TASK_CONCURRENCY}"

${QSUB_BIN} \
  -t "1-${TASKS_TO_RUN}" \
  -tc "${TASK_CONCURRENCY}" \
  -v "PIPELINE_CFG=${PIPELINE_CFG},NETWORK_DIR=${NETWORK_DIR},NETWORK_GLOB=${NETWORK_GLOB},MANIFEST_PATH=${MANIFEST_PATH},AUTO_BUILD_MANIFEST=${AUTO_BUILD_MANIFEST}" \
  "${ROOT_DIR}/run_pipeline.sh"

