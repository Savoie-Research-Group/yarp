#!/bin/bash
# file: monitor_submit.sh
# Monitors queue and submits job chunks when capacity is available
#
# Usage:
#   ./monitor_submit.sh              # Run in foreground
#   tmux new-session -d -s submit './monitor_submit.sh'  # Background
#
# Reads job_*.txt files (each containing up to 1000 jobs) and submits them
# one at a time via condor_submit, waiting for queue capacity between chunks.

# --- CONFIGURATION ---
MAX_JOBS=10000
USER_ID="${USER}"
CHECK_INTERVAL=300  # 5 minutes
SUBMIT_FILE="batch_job.submit"
FILE_PATTERN="job_*.txt"
# ---------------------

echo "=============================================="
echo "DFT Energy — Job Submission Monitor"
echo "=============================================="
echo "Submit file:    $SUBMIT_FILE"
echo "File pattern:   $FILE_PATTERN"
echo "Max jobs:       $MAX_JOBS"
echo "User:           $USER_ID"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Create tracking folder
mkdir -p submitted_batches

# Get current job count
get_job_count() {
    local count
    count=$(condor_q "$USER_ID" 2>/dev/null | grep "Total for query" | awk '{print $4}')

    if [ -z "$count" ]; then
        count=$(condor_q "$USER_ID" 2>/dev/null | awk -v user="$USER_ID" '$0 ~ user {sum += $10} END {print sum+0}')
    fi

    echo "${count:-0}"
}

# Check if submit file exists
if [ ! -f "$SUBMIT_FILE" ]; then
    echo "Error: Submit file not found: $SUBMIT_FILE"
    exit 1
fi

# Count pending job files
PENDING_FILES=$(ls $FILE_PATTERN 2>/dev/null | wc -l)
if [ "$PENDING_FILES" -eq 0 ]; then
    echo "Error: No job files found matching $FILE_PATTERN"
    exit 1
fi
echo "Job files to submit: $PENDING_FILES"
echo ""

# Loop through job files (sorted numerically)
for batch_file in $(ls $FILE_PATTERN 2>/dev/null | sort -t_ -k2 -n); do
    # Check if file exists
    [ -f "$batch_file" ] || continue

    # Skip if already submitted (in case of restart)
    if [ -f "submitted_batches/$batch_file" ]; then
        echo "Skipping $batch_file (already submitted)"
        continue
    fi

    # Monitor loop — wait until queue has capacity
    while true; do
        current_jobs=$(get_job_count)

        echo "[$(date '+%H:%M:%S')] Queue: $current_jobs / $MAX_JOBS jobs"

        if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
            echo "   -> Queue has capacity. Submitting..."
            break
        else
            echo "   -> Queue full. Waiting $((CHECK_INTERVAL / 60)) minutes..."
            sleep $CHECK_INTERVAL
        fi
    done

    # Submit the chunk
    echo ""
    echo "=============================================="
    echo "Submitting: $batch_file"
    echo "=============================================="

    JOB_COUNT=$(wc -l < "$batch_file")
    echo "Jobs in file: $JOB_COUNT"

    condor_submit "$SUBMIT_FILE" input_list="$batch_file"
    SUBMIT_EXIT=$?

    if [ $SUBMIT_EXIT -eq 0 ]; then
        # Move to submitted folder
        mv "$batch_file" submitted_batches/
        echo "Moved $batch_file to submitted_batches/"
    else
        echo "Warning: condor_submit returned exit code $SUBMIT_EXIT"
        echo "Will retry $batch_file on next cycle"
        sleep 60
        continue
    fi

    echo ""

    # Short pause to let the scheduler digest
    sleep 10
done

echo "=============================================="
echo "All job files submitted!"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  condor_q $USER_ID"
echo "  ./status.sh"
