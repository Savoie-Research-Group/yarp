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
# By default, build manifest once at submit time and skip rebuild inside each task.
RUN_TASK_AUTO_BUILD_MANIFEST="${RUN_TASK_AUTO_BUILD_MANIFEST:-0}"
QSUB_BIN="${QSUB_BIN:-qsub}"
# Optional cap on number of manifest entries to submit. Empty/0 means no cap.
ARRAY_MAX_TASKS="${ARRAY_MAX_TASKS:-0}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-1}"
TARGET_NETWORKS="${TARGET_NETWORKS:-}"
START_TASK="${START_TASK:-1}"
SLOTS_PER_TASK="${SLOTS_PER_TASK:-4}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/job_logs}"

# Optional runtime override passed through to run_pipeline.sh:
#   OVERRIDE_OUTPUT_MODE=debug|subnetwork|production
OVERRIDE_OUTPUT_MODE="${OVERRIDE_OUTPUT_MODE:-production}"
# Optional runtime output root override (useful for separating debug vs production artifacts).
OVERRIDE_OUTPUT_DIR="${OVERRIDE_OUTPUT_DIR:-}"

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

if [[ "${START_TASK}" -lt 1 ]]; then
  START_TASK=1
fi

if [[ "${START_TASK}" -gt "${TOTAL_TASKS}" ]]; then
  echo "START_TASK=${START_TASK} exceeds manifest length (${TOTAL_TASKS}); nothing to submit."
  exit 0
fi

if [[ -n "${TARGET_NETWORKS}" ]] && [[ "${TARGET_NETWORKS}" =~ ^[0-9]+$ ]] && [[ "${TARGET_NETWORKS}" -gt 0 ]]; then
  TASKS_TO_RUN="${TARGET_NETWORKS}"
elif [[ "${ARRAY_MAX_TASKS}" =~ ^[0-9]+$ ]] && [[ "${ARRAY_MAX_TASKS}" -gt 0 ]] && [[ "${ARRAY_MAX_TASKS}" -lt "${TOTAL_TASKS}" ]]; then
  TASKS_TO_RUN="${ARRAY_MAX_TASKS}"
else
  TASKS_TO_RUN="${TOTAL_TASKS}"
fi

END_TASK=$(( START_TASK + TASKS_TO_RUN - 1 ))
if [[ "${END_TASK}" -gt "${TOTAL_TASKS}" ]]; then
  END_TASK="${TOTAL_TASKS}"
fi

if [[ "${SLOTS_PER_TASK}" -lt 1 ]]; then
  SLOTS_PER_TASK=1
fi

echo "Submitting UGE array:"
echo "  pipeline_cfg=${PIPELINE_CFG}"
echo "  manifest=${MANIFEST_PATH}"
echo "  total_manifest_tasks=${TOTAL_TASKS}"
echo "  array_task_range=${START_TASK}-${END_TASK}"
echo "  task_concurrency=${TASK_CONCURRENCY}"
echo "  slots_per_task=${SLOTS_PER_TASK}"
echo "  target_networks_override=${TARGET_NETWORKS:-<none>}"
echo "  array_max_tasks_cap=${ARRAY_MAX_TASKS}"
echo "  override_output_mode=${OVERRIDE_OUTPUT_MODE:-<none>}"
echo "  override_output_dir=${OVERRIDE_OUTPUT_DIR:-<none>}"
echo "  run_task_auto_build_manifest=${RUN_TASK_AUTO_BUILD_MANIFEST}"
echo "  log_dir=${LOG_DIR}"

mkdir -p "${LOG_DIR}"

${QSUB_BIN} \
  -o "${LOG_DIR}" \
  -e "${LOG_DIR}" \
  -t "${START_TASK}-${END_TASK}" \
  -tc "${TASK_CONCURRENCY}" \
  -pe smp "${SLOTS_PER_TASK}" \
  -v "PIPELINE_CFG=${PIPELINE_CFG},NETWORK_DIR=${NETWORK_DIR},NETWORK_GLOB=${NETWORK_GLOB},MANIFEST_PATH=${MANIFEST_PATH},AUTO_BUILD_MANIFEST=${RUN_TASK_AUTO_BUILD_MANIFEST},OVERRIDE_OUTPUT_MODE=${OVERRIDE_OUTPUT_MODE},OVERRIDE_OUTPUT_DIR=${OVERRIDE_OUTPUT_DIR},LOG_DIR=${LOG_DIR}" \
  "${ROOT_DIR}/run_pipeline.sh"
