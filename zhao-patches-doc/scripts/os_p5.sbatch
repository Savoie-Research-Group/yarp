#!/bin/bash
#SBATCH --job-name=os_p5
#SBATCH --output=Scripts/v2/os_test_final/logs/os_p5_%A_%a.out
#SBATCH --error=Scripts/v2/os_test_final/logs/os_p5_%A_%a.err
#SBATCH -A bsavoie
#SBATCH --partition=cpu
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time 04:00:00
#SBATCH --array=1-100%1000
#SBATCH --requeue

set -euo pipefail
cd /scratch/negishi/li1724/SI-Downloads/SI_Agent/doi_tar_zsts

PY=/home/li1724/.conda/envs/2022.10-py39/copy-classy-yarp/bin/python
export NEW_YARP_PATH=/home/li1724/061226-YARP-again/Zhao-YARP/classy-yarp-final

SHARD=$(printf "Scripts/v2/os_test_final/shards_p5/shard_%03d.txt" "$SLURM_ARRAY_TASK_ID")
OUT=$(printf   "Scripts/v2/os_test_final/results_p5/shard_%03d.csv" "$SLURM_ARRAY_TASK_ID")

if [ ! -f "$SHARD" ]; then
    echo "no shard ${SLURM_ARRAY_TASK_ID}"
    exit 0
fi

EXPECTED=$(( $(wc -l < "$SHARD") + 1 ))
if [ -s "$OUT" ]; then
    HAVE=$(wc -l < "$OUT")
    if [ "$HAVE" -eq "$EXPECTED" ]; then
        echo "skip ${SLURM_ARRAY_TASK_ID}: complete (${HAVE} rows)"
        exit 0
    fi
    echo "partial (${HAVE}/${EXPECTED}), restart"
    rm -f "$OUT"
fi

echo "=== p5-shard ${SLURM_ARRAY_TASK_ID} start $(date) host=$(hostname) ==="
"$PY" Scripts/v2/os_test_new_yarp/os_new_yarp_shard.py "$SHARD" --out "$OUT"
echo "=== p5-shard ${SLURM_ARRAY_TASK_ID} end $(date) ==="
