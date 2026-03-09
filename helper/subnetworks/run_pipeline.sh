#!/bin/bash
# ==============================
# Job-array runner for subnetwork kinetics pipeline
# Submit with an explicit task range, e.g.:
#   qsub -t 1-10 run_pipeline.sh
# ==============================

#$ -N Kinetics_Pipeline
#$ -cwd
#$ -V
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -t 1-1
#$ -tc 1

set -euo pipefail

module load conda

conda activate yarp-again

# Ensure matplotlib/font cache is writable on cluster nodes.
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl_cache_subnetwork_kinetics}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/xdg_cache_subnetwork_kinetics}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

# Under UGE/SGE, this script may be copied to a spool directory before execution.
# Prefer the original submit directory so relative repo paths still work.
if [[ -n "${SGE_O_WORKDIR:-}" ]]; then
  ROOT_DIR="${SGE_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SCRIPT_DIR="${ROOT_DIR}/pipeline"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-/users/tburton2/.conda/envs/yarp-again}/bin/python}"
PIPELINE_CFG="${PIPELINE_CFG:-${ROOT_DIR}/pipeline/configs/pipeline_config.yaml}"
NETWORK_DIR="${NETWORK_DIR:-${ROOT_DIR}/networks}"
NETWORK_GLOB="${NETWORK_GLOB:-*.pkl}"
MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/networks/manifest.txt}"
AUTO_BUILD_MANIFEST="${AUTO_BUILD_MANIFEST:-1}"
RUNTIME_CFG="${RUNTIME_CFG:-}"
OVERRIDE_OUTPUT_MODE="${OVERRIDE_OUTPUT_MODE:-production}"
OVERRIDE_OUTPUT_DIR="${OVERRIDE_OUTPUT_DIR:-}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/job_logs}"
LOG_CLEAN_DIR="${LOG_CLEAN_DIR:-${LOG_DIR}}"
LOG_CLEAN_HOURS="${LOG_CLEAN_HOURS:-4}"
LOG_END_MARKER="${LOG_END_MARKER:-TIME_ENDED:}"
RUNTIME_CFG_AUTO=0

mkdir -p "${LOG_DIR}"

on_exit() {
  local exit_code=$?
  echo "${LOG_END_MARKER} $(date -u +%Y-%m-%dT%H:%M:%SZ) JOB_ID=${JOB_ID:-unknown} TASK_ID=${SGE_TASK_ID:-unknown} EXIT_CODE=${exit_code}"
  if [[ "${RUNTIME_CFG_AUTO:-0}" == "1" ]] && [[ -n "${RUNTIME_CFG:-}" ]] && [[ -f "${RUNTIME_CFG}" ]]; then
    rm -f "${RUNTIME_CFG}" || true
  fi
  if [[ -x "${SCRIPT_DIR}/cleanup_completed_logs.sh" ]]; then
    "${SCRIPT_DIR}/cleanup_completed_logs.sh" "${LOG_CLEAN_DIR}" "${LOG_CLEAN_HOURS}" "${LOG_END_MARKER}" || true
  fi
}
trap on_exit EXIT

# Avoid shared runtime-config races across array tasks/jobs unless explicitly set.
if [[ -z "${RUNTIME_CFG}" ]]; then
  JOB_TAG="${JOB_ID:-nojob}"
  TASK_TAG="${SGE_TASK_ID:-notask}"
  TMP_CFG_DIR="${TMPDIR:-/tmp}"
  mkdir -p "${TMP_CFG_DIR}"
  RUNTIME_CFG="${TMP_CFG_DIR}/subnetwork_pipeline_runtime.${JOB_TAG}.${TASK_TAG}.yaml"
  RUNTIME_CFG_AUTO=1
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import sys; import yarp; print('sys.executable=', sys.executable); print('yarp=', yarp.__file__)"

if [[ ! -f "${SCRIPT_DIR}/run_subnetwork_pipeline.py" ]]; then
  echo "Error: pipeline entrypoint not found at ${SCRIPT_DIR}/run_subnetwork_pipeline.py"
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "SGE_O_WORKDIR=${SGE_O_WORKDIR:-<unset>}"
  exit 1
fi

if [[ "${AUTO_BUILD_MANIFEST}" == "1" ]]; then
  mkdir -p "$(dirname "${MANIFEST_PATH}")"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
network_dir = Path(r"""${NETWORK_DIR}""").expanduser().resolve()
glob_pat = r"""${NETWORK_GLOB}"""
manifest_path = Path(r"""${MANIFEST_PATH}""").expanduser().resolve()
if not network_dir.exists():
    raise FileNotFoundError(f"Network directory does not exist: {network_dir}")
paths = sorted([p.resolve() for p in network_dir.glob(glob_pat) if p.is_file()])
if not paths:
    raise RuntimeError(f"No network files found in {network_dir} matching {glob_pat}")
manifest_path.write_text("\\n".join(str(p) for p in paths) + "\\n")
print(f"Manifest written: {manifest_path} | entries={len(paths)}")
PY
fi

mkdir -p "$(dirname "${RUNTIME_CFG}")"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml
cfg_path = Path(r"""${PIPELINE_CFG}""").expanduser().resolve()
manifest_path = Path(r"""${MANIFEST_PATH}""").expanduser().resolve()
runtime_cfg = Path(r"""${RUNTIME_CFG}""").expanduser().resolve()
override_output_mode = r"""${OVERRIDE_OUTPUT_MODE}""".strip()
override_output_dir = r"""${OVERRIDE_OUTPUT_DIR}""".strip()
cfg = yaml.safe_load(cfg_path.read_text()) or {}
if not isinstance(cfg, dict):
    raise RuntimeError(f"Pipeline config must be a mapping: {cfg_path}")

cfg["network_manifest"] = str(manifest_path)
if override_output_mode:
    cfg["output_mode"] = override_output_mode
if override_output_dir:
    cfg["output_dir"] = override_output_dir

runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Runtime config written: {runtime_cfg}")
print(f"Runtime manifest: {manifest_path}")
print(f"Configured output_mode={cfg.get('output_mode')}")
print(f"Configured max_products_per_network={cfg.get('max_products_per_network')}")
print(f"Configured output_dir={cfg.get('output_dir')}")
PY

cd "$ROOT_DIR"
"$PYTHON_BIN" "${SCRIPT_DIR}/run_subnetwork_pipeline.py" --config "$RUNTIME_CFG"
