#!/bin/bash
# DFT-style submission monitor for YARP OSG batches.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${1:-$PWD}"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
STATE_DIR="$WORK_DIR/.yarp_osg"
SUBMIT_FILE="$STATE_DIR/batch_egat.submit"
MAX_JOBS="${YARP_OSG_MAX_JOBS:-10000}"
CHECK_INTERVAL="${YARP_OSG_CHECK_INTERVAL:-300}"
USER_ID="${USER}"

mkdir -p "$STATE_DIR/submitted_batches"

get_job_count() {
    local count
    count=$(condor_q "$USER_ID" 2>/dev/null | grep "Total for query" | awk '{print $4}')
    echo "${count:-0}"
}

if [ ! -f "$SUBMIT_FILE" ]; then
    echo "Error: submit file not found: $SUBMIT_FILE"
    exit 1
fi

echo "=============================================="
echo "YARP OSG submit monitor"
echo "=============================================="
echo "Work dir: $WORK_DIR"
echo "Submit:   $SUBMIT_FILE"
echo "Max jobs: $MAX_JOBS"
echo ""

for batch_file in $(find "$STATE_DIR" -maxdepth 1 -name 'job_*.tsv' -type f | sort -V); do
    base="$(basename "$batch_file")"
    if [ -f "$STATE_DIR/submitted_batches/$base" ]; then
        echo "Skipping $base (already submitted)"
        continue
    fi

    while true; do
        current_jobs="$(get_job_count)"
        echo "[$(date '+%H:%M:%S')] Queue: $current_jobs / $MAX_JOBS"
        if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep "$CHECK_INTERVAL"
    done

    echo "Submitting $base"
    output="$(condor_submit "$SUBMIT_FILE" "input_list=$batch_file" 2>&1)"
    exit_code=$?
    echo "$output"
    if [ "$exit_code" -ne 0 ]; then
        echo "condor_submit failed for $base; will retry later"
        sleep 60
        continue
    fi

    cluster="$(echo "$output" | sed -nE 's/.*cluster ([0-9]+).*/\1/p' | tail -1)"
    if [ -z "$cluster" ]; then
        echo "Could not parse cluster id from condor_submit output"
        exit 1
    fi

    "$SCRIPT_DIR/yarp-osg" record-submit "$WORK_DIR" --cluster-id "$cluster" --batch-file "$batch_file"
    mv "$batch_file" "$STATE_DIR/submitted_batches/"
    sleep 10
done

echo "All YARP OSG batch files submitted."
