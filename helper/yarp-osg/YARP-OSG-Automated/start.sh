#!/bin/bash
# One-button launcher for the YARP EGAT OSG milestone.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${1:-$PWD}"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
STATE_DIR="$WORK_DIR/.yarp_osg"

MAX_JOBS="${YARP_OSG_MAX_JOBS:-10000}"

echo "=============================================="
echo "YARP OSG EGAT workflow"
echo "=============================================="
echo "Workflow dir: $SCRIPT_DIR"
echo "Work dir:     $WORK_DIR"
echo "Max jobs:     $MAX_JOBS"
echo ""

echo "[1/5] Checking YARP state"
if ! ls "$WORK_DIR"/*.json >/dev/null 2>&1; then
    echo "Error: no STATUS JSON file found in $WORK_DIR"
    exit 1
fi

echo "[2/5] Creating OSG directories"
mkdir -p "$STATE_DIR/logs" "$STATE_DIR/submitted_batches" "$STATE_DIR/tasks"

echo "[3/5] Preparing EGAT inputs and submit artifacts"
"$SCRIPT_DIR/yarp-osg" prepare-egat "$WORK_DIR" --workflow-dir "$SCRIPT_DIR"

echo "[4/5] Checking generated batches"
PENDING="$(find "$STATE_DIR" -maxdepth 1 -name 'job_*.tsv' -type f | wc -l | tr -d ' ')"
echo "Pending batch files: $PENDING"
if [ "$PENDING" -eq 0 ]; then
    echo "No EGAT batches to submit."
    exit 0
fi

echo "[5/5] Starting monitors"
if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t yarp_osg_submit 2>/dev/null || true
    tmux kill-session -t yarp_osg_health 2>/dev/null || true
    tmux new-session -d -s yarp_osg_submit "cd '$SCRIPT_DIR' && ./monitor_submit.sh '$WORK_DIR'"
    tmux new-session -d -s yarp_osg_health "cd '$SCRIPT_DIR' && ./monitor_health.sh '$WORK_DIR'"
    echo "Started tmux sessions: yarp_osg_submit, yarp_osg_health"
else
    echo "tmux not found; run these manually:"
    echo "  $SCRIPT_DIR/monitor_submit.sh $WORK_DIR"
    echo "  $SCRIPT_DIR/monitor_health.sh $WORK_DIR"
fi

echo ""
echo "Status:"
echo "  $SCRIPT_DIR/status.sh $WORK_DIR"
echo "Harvest after completion:"
echo "  $SCRIPT_DIR/yarp-osg harvest $WORK_DIR"
