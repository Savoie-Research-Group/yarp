#!/bin/bash
# file: status.sh
# Quick status check for DFT Energy workflow
#
# Usage:
#   ./status.sh

USER_ID="${USER}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 DFT Energy — Workflow Status                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Job queue status
echo "📊 Queue Status"
echo "───────────────────────────────────────"
if command -v condor_q &> /dev/null; then
    condor_q "$USER_ID" 2>/dev/null | tail -3
else
    echo "  (condor_q not available)"
fi
echo ""

# Submission progress
echo "📁 Submission Progress"
echo "───────────────────────────────────────"

PENDING=$(ls -1 job_*.txt 2>/dev/null | wc -l)
SUBMITTED=$(ls -1 submitted_batches/job_*.txt 2>/dev/null | wc -l)

echo "  Chunks pending:    $PENDING"
echo "  Chunks submitted:  $SUBMITTED"

if [ -f "file_list.txt" ]; then
    TOTAL_JOBS=$(wc -l < file_list.txt)
    echo "  Total jobs:        $TOTAL_JOBS"
fi
echo ""

# Held jobs summary
echo "⚠️  Held Jobs"
echo "───────────────────────────────────────"
if command -v condor_q &> /dev/null; then
    HELD_COUNT=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -format "%d\n" ClusterId 2>/dev/null | wc -l)
    if [ "$HELD_COUNT" -gt 0 ]; then
        echo "  Total held: $HELD_COUNT"
        echo ""
        echo "  Top hold reasons:"
        condor_q "$USER_ID" -constraint 'JobStatus == 5' -format "%s\n" HoldReason 2>/dev/null | \
            sort | uniq -c | sort -rn | head -5 | \
            while read -r count reason; do
                reason_short=$(echo "$reason" | cut -c1-50)
                printf "    %4d — %s\n" "$count" "$reason_short"
            done
    else
        echo "  None! 🎉"
    fi
else
    echo "  (condor_q not available)"
fi
echo ""

# Maxed-out jobs
if [ -f "held_jobs_maxed.log" ] && [ -s "held_jobs_maxed.log" ]; then
    MAXED_COUNT=$(wc -l < held_jobs_maxed.log)
    echo "🛑 Maxed-Out Jobs (need manual review)"
    echo "───────────────────────────────────────"
    echo "  Total: $MAXED_COUNT"
    echo "  Log: held_jobs_maxed.log"
    echo "  (These hit 40GB/10CPU and were NOT auto-released)"
    echo ""
fi

# Results status
echo "📦 Results"
echo "───────────────────────────────────────"
if [ -d "results" ]; then
    RESULT_COUNT=$(find results -name "*_results.tar.gz" -type f 2>/dev/null | wc -l)
    echo "  Local result files: $RESULT_COUNT"
else
    echo "  (no results/ directory yet)"
fi

# Backup status
DEST_BASE="/ospool/ap40/data/$USER"
PROJECT_PARENT=$(basename "$(dirname "$(pwd)")")
BACKUP_DIR="$DEST_BASE/dft_energy_backups/$PROJECT_PARENT"
if [ -d "$BACKUP_DIR" ]; then
    BACKUP_COUNT=$(ls "$BACKUP_DIR"/*.tar.gz 2>/dev/null | wc -l)
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
    echo "  Backups:            $BACKUP_COUNT archives ($BACKUP_SIZE)"
fi
echo ""

# Running processes
echo "🔄 Background Processes"
echo "───────────────────────────────────────"
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux list-sessions 2>/dev/null | grep -E "submit|health|backup" || echo "")
    if [ -n "$SESSIONS" ]; then
        echo "$SESSIONS" | while read -r line; do
            echo "  $line"
        done
    else
        echo "  No monitor sessions running"
    fi
else
    echo "  (tmux not available)"
fi
echo ""

# Resource tracking
if [ -f "resource_tracking.dat" ] && [ -s "resource_tracking.dat" ]; then
    TRACKED=$(wc -l < resource_tracking.dat)
    echo "📈 Resource Escalation"
    echo "───────────────────────────────────────"
    echo "  Jobs with upgrades: $TRACKED"
    echo "  Levels: L0(12GB/4) → L1(20GB/6) → L2(32GB/8) → L3(40GB/10)"
    echo ""
fi

echo "Quick commands:"
echo "  condor_q $USER_ID              # Full queue status"
echo "  tmux attach -t submit          # View submission monitor"
echo "  tmux attach -t health          # View health monitor"
echo "  tmux attach -t backup          # View backup monitor"
echo ""
