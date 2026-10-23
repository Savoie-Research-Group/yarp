#!/bin/bash
# Job-array template for step 02. submit.py reads this file, fills in the
# double-brace placeholders and writes RUN_DIR/job.sh; it never runs qsub.
#
# Written for SGE (Notre Dame CRC). For another system, edit the scheduler
# directives, the environment setup and the task-ID variable below, but keep
# the placeholders: JOB_NAME, N_SHARDS, LOG_DIR, PYTHON, RUN_SHARD, RUN_DIR.
# One array task runs one shard of RUN_DIR/rows.csv.
#$ -N {{JOB_NAME}}
#$ -t 1-{{N_SHARDS}}
#$ -q long
#$ -pe smp 1
#$ -l h_rt=04:00:00
#$ -j y
#$ -o {{LOG_DIR}}/
set -eu

# Environment: one thread per task. Add module loads or conda activation here
# if the python below needs them.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SHARD=$SGE_TASK_ID

echo "host $(hostname) shard $SHARD start $(date '+%F %T')"
"{{PYTHON}}" "{{RUN_SHARD}}" --rows "{{RUN_DIR}}/rows.csv" --out "{{RUN_DIR}}/out" --shard "$SHARD"
echo "end $(date '+%F %T')"
