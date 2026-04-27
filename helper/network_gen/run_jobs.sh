#!/bin/bash
# ==============================
# Job array reading from joblist.txt
# ==============================

#$ -N YARPagain_Network_Gen
#$ -cwd
#$ -V
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:00:00
#$ -t 1-100
#$ -tc 100

set -euo pipefail

conda activate yarp-again

BASE_DIR="${SGE_O_WORKDIR:-${PWD}}"
RUN_LIST="${RUN_LIST:-${BASE_DIR}/outputs/joblist.txt}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${BASE_DIR}/network_gen.py}"

if [[ ! -f "${RUN_LIST}" ]]; then
  echo "ERROR: RUN_LIST not found: ${RUN_LIST}" >&2
  echo "Set RUN_LIST explicitly via qsub -v RUN_LIST=/abs/path/joblist.txt" >&2
  exit 1
fi

TASK_ID="${SGE_TASK_ID:-${1:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: TASK_ID not set. Use qsub -t 1-N run_jobs.sh or pass task id as arg." >&2
  exit 1
fi

RUN_DIR="$(sed -n "${TASK_ID}p" "${RUN_LIST}")"

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: Task ${TASK_ID}: no directory found in ${RUN_LIST}" >&2
  exit 1
fi

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${BASE_DIR}/${RUN_DIR}"
fi

RUN_NAME="$(basename "${RUN_DIR}")"
CONFIG_PATH="${RUN_DIR}/config.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Missing config for task ${TASK_ID}: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "======================================================="
echo "Task ${TASK_ID} -> ${RUN_NAME}"
echo "RUN_DIR: ${RUN_DIR}"
echo "RUN_LIST: ${RUN_LIST}"
echo "Host: $(hostname)"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Config: ${CONFIG_PATH}"
echo "======================================================="

echo "=========Running network generation...========="

cd "${RUN_DIR}"

python "$PYTHON_SCRIPT" --config "$CONFIG_PATH" > run.out 2> run.err

echo " Network Generation Completed for ${RUN_NAME}"
echo "Output: ${RUN_DIR}/run.out"
echo "Errors: ${RUN_DIR}/run.err" 
echo "SMILES String: $(head -n 1 smi.txt)"
echo "======================================================="

