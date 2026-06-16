#!/bin/bash
# Static EGAT worker for HTCondor container_image jobs.

set -uo pipefail

TASK_ID="${1:-unknown}"
ATTEMPT="${2:-0}"
WORKDIR="${_CONDOR_SCRATCH_DIR:-$PWD}"
START_EPOCH="$(date +%s)"

cd "$WORKDIR"

touch forward.log forward.err reverse.log reverse.err forward_out.csv reverse_out.csv

echo "hostname=$(hostname)"
echo "workdir=$PWD"
echo "task_id=$TASK_ID"
echo "attempt=$ATTEMPT"
echo "_CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-}"
echo "OSG_SITE_NAME=${OSG_SITE_NAME:-}"
echo "GLIDEIN_Site=${GLIDEIN_Site:-}"

write_result() {
    status="$1"
    exit_code="$2"
    category="$3"
    message="$4"
    end_epoch="$(date +%s)"
    runtime="$((end_epoch - START_EPOCH))"
    present_forward=false
    present_reverse=false
    [ -s forward_out.csv ] && present_forward=true
    [ -s reverse_out.csv ] && present_reverse=true
    cat > task_result.json <<JSON
{
  "task_id": "$TASK_ID",
  "reaction_id": null,
  "task_type": "egat_ml_predict",
  "attempt": $ATTEMPT,
  "exit_code": $exit_code,
  "status": "$status",
  "hostname": "$(hostname)",
  "start_time_epoch": $START_EPOCH,
  "end_time_epoch": $end_epoch,
  "runtime_seconds": $runtime,
  "expected_outputs": ["forward_out.csv", "reverse_out.csv"],
  "present_outputs": {
    "forward_out.csv": $present_forward,
    "reverse_out.csv": $present_reverse
  },
  "error_category": "$category",
  "error_message": "$message"
}
JSON
}

if [ ! -s egat_command.txt ]; then
    echo "Missing egat_command.txt" >&2
    write_result "failed" 64 "configuration" "missing_egat_command"
    exit 64
fi

EGAT_COMMAND="$(head -n 1 egat_command.txt)"
if [ -z "$EGAT_COMMAND" ]; then
    echo "EGAT command is empty" >&2
    write_result "failed" 64 "configuration" "empty_egat_command"
    exit 64
fi

if [ ! -f forward_in.csv ] || [ ! -f reverse_in.csv ]; then
    echo "Missing EGAT input CSV" >&2
    write_result "failed" 65 "configuration" "missing_input_csv"
    exit 65
fi

echo "Running forward EGAT command"
bash -lc "$EGAT_COMMAND --input forward_in.csv --output forward_out.csv" > forward.log 2> forward.err
FWD_EXIT=$?

echo "Running reverse EGAT command"
bash -lc "$EGAT_COMMAND --input reverse_in.csv --output reverse_out.csv" > reverse.log 2> reverse.err
REV_EXIT=$?

if [ "$FWD_EXIT" -ne 0 ] || [ "$REV_EXIT" -ne 0 ]; then
    write_result "failed" 20 "chemistry" "egat_command_failed"
    exit 20
fi

if [ ! -s forward_out.csv ] || [ ! -s reverse_out.csv ]; then
    write_result "failed" 21 "missing_output" "missing_egat_output_csv"
    exit 21
fi

write_result "success" 0 "" ""
exit 0
