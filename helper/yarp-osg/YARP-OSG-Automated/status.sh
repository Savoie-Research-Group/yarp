#!/bin/bash
# Quick YARP OSG status report.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${1:-$PWD}"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
STATE_DIR="$WORK_DIR/.yarp_osg"
USER_ID="${USER}"

echo "=============================================="
echo "YARP OSG status"
echo "=============================================="
echo "Work dir: $WORK_DIR"
echo ""

"$SCRIPT_DIR/yarp-osg" status "$WORK_DIR"
echo ""

echo "Queue status"
echo "------------"
if command -v condor_q >/dev/null 2>&1; then
    condor_q "$USER_ID" 2>/dev/null | tail -3
else
    echo "(condor_q not available)"
fi
echo ""

echo "Submission progress"
echo "-------------------"
PENDING="$(find "$STATE_DIR" -maxdepth 1 -name 'job_*.tsv' -type f 2>/dev/null | wc -l | tr -d ' ')"
SUBMITTED="$(find "$STATE_DIR/submitted_batches" -maxdepth 1 -name 'job_*.tsv' -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "Pending batches:   ${PENDING:-0}"
echo "Submitted batches: ${SUBMITTED:-0}"

echo ""
echo "Held jobs"
echo "---------"
if command -v condor_q >/dev/null 2>&1; then
    condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | head -10
else
    echo "(condor_q not available)"
fi

echo ""
echo "Commands"
echo "--------"
echo "Harvest: $SCRIPT_DIR/yarp-osg harvest $WORK_DIR"
echo "Advance: $SCRIPT_DIR/yarp-osg advance $WORK_DIR"
