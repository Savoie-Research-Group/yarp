#!/bin/bash
# file: add_data.sh
# Add new input data to a running DFT workflow
#
# Usage:
#   ./add_data.sh                       # Process all new *.tar.gz in current dir
#   ./add_data.sh /path/to/new.tar.gz   # Process specific archive(s)
#
# What it does:
#   1. Extracts new tar.gz → inputs/ (preserving nested structure)
#   2. Scans for .zip files NOT already in file_list.txt
#   3. Appends new entries to file_list.txt
#   4. Pre-creates mirrored directories under results/
#   5. Splits new entries into numbered job_*.txt chunks (continues numbering)
#   6. Restarts only the submit monitor (health & backup stay alive)
#
# Safe to run multiple times — it never re-adds zips already tracked.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== CONFIGURATION ========================
JOBS_PER_CHUNK=1000
# ===============================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              DFT Energy — Add New Input Data                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Determine which tar.gz files to process
# -----------------------------------------------------------------------------
if [ $# -gt 0 ]; then
    # Specific files provided as arguments
    INPUT_TARBALLS="$@"
else
    # Auto-detect new tar.gz files in current directory
    # Exclude backup archives and results packages
    INPUT_TARBALLS=$(ls *.tar.gz 2>/dev/null | grep -v 'results_package' | grep -v 'backup_' || true)
fi

if [ -z "$INPUT_TARBALLS" ]; then
    echo -e "${RED}Error: No *.tar.gz files found.${NC}"
    echo ""
    echo "Usage:"
    echo "  ./add_data.sh                       # All *.tar.gz in current dir"
    echo "  ./add_data.sh /path/to/new.tar.gz   # Specific file(s)"
    exit 1
fi

echo -e "Archives to process:"
for t in $INPUT_TARBALLS; do
    echo -e "  ${GREEN}$(basename "$t")${NC}"
done
echo ""

# -----------------------------------------------------------------------------
# Preflight: check that workflow has been initialized
# -----------------------------------------------------------------------------
if [ ! -f "file_list.txt" ]; then
    echo -e "${RED}Error: file_list.txt not found. Run start.sh first to initialize the workflow.${NC}"
    exit 1
fi

if [ ! -d "inputs" ] || [ ! -d "results" ]; then
    echo -e "${RED}Error: inputs/ or results/ directory missing. Run start.sh first.${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: Extract new archives
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5]${NC} Extracting new input archives..."

for TARBALL in $INPUT_TARBALLS; do
    if [ ! -f "$TARBALL" ]; then
        echo -e "  ${YELLOW}Warning:${NC} $TARBALL not found, skipping"
        continue
    fi
    echo "  Extracting: $(basename "$TARBALL")"
    tar -xzf "$TARBALL" -C inputs/
done

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Find NEW zip files not already tracked
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/5]${NC} Scanning for new .zip files..."

INPUTS_ABS="$(cd inputs && pwd)"

# Build sorted list of all zip absolute paths currently on disk
ALL_ZIPS_FILE=".all_zips_$$.txt"
find "$INPUTS_ABS" -name "*.zip" -type f | sort > "$ALL_ZIPS_FILE"
TOTAL_ON_DISK=$(wc -l < "$ALL_ZIPS_FILE")
echo "  Zips on disk:       $TOTAL_ON_DISK"

# Build sorted list of already-tracked absolute paths (column 1 of file_list.txt)
TRACKED_ZIPS_FILE=".tracked_zips_$$.txt"
awk -F', ' '{print $1}' file_list.txt | sort > "$TRACKED_ZIPS_FILE"
EXISTING_COUNT=$(wc -l < "$TRACKED_ZIPS_FILE")
echo "  Previously tracked: $EXISTING_COUNT"

# Fast set difference: zips on disk but NOT in file_list.txt
NEW_ZIPS_FILE=".new_zips_$$.txt"
comm -23 "$ALL_ZIPS_FILE" "$TRACKED_ZIPS_FILE" > "$NEW_ZIPS_FILE"
NEW_COUNT=$(wc -l < "$NEW_ZIPS_FILE")

# Clean up temp files
rm -f "$ALL_ZIPS_FILE" "$TRACKED_ZIPS_FILE"

if [ "$NEW_COUNT" -eq 0 ]; then
    echo -e "  ${YELLOW}No new .zip files found.${NC}"
    echo "  All zips in inputs/ are already tracked in file_list.txt."
    rm -f "$NEW_ZIPS_FILE"
    exit 0
fi

echo -e "  New .zip files:     ${GREEN}$NEW_COUNT${NC}"

# Convert new zip paths to file_list format: "abspath, relpath_no_ext"
NEW_LIST_FILE=".new_entries_$$.txt"
while IFS= read -r zipfile; do
    relpath="${zipfile#$INPUTS_ABS/}"
    relpath_no_ext="${relpath%.zip}"
    echo "$zipfile, $relpath_no_ext"
done < "$NEW_ZIPS_FILE" > "$NEW_LIST_FILE"

rm -f "$NEW_ZIPS_FILE"

echo ""

# -----------------------------------------------------------------------------
# Step 3: Append new entries to file_list.txt and mirror directories
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5]${NC} Updating file_list.txt and mirroring directories..."

# Append to master file list
cat "$NEW_LIST_FILE" >> file_list.txt
TOTAL_COUNT=$(wc -l < file_list.txt)
echo "  Appended $NEW_COUNT entries → file_list.txt now has $TOTAL_COUNT total"

# Mirror any new directories under results/
(cd inputs && find . -type d -exec mkdir -p "../results/{}" \;)
echo "  Mirrored directory structure under results/"

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Split new entries into numbered chunks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5]${NC} Creating job chunks for new entries..."

# Find the highest existing chunk number (including submitted ones)
MAX_NUM=0
ALL_JOB_FILES=$(ls job_*.txt submitted_batches/job_*.txt 2>/dev/null || true)
for f in $ALL_JOB_FILES; do
    [ -f "$f" ] || continue
    num=$(basename "$f" | sed -n 's/^job_\([0-9]\+\)\.txt$/\1/p')
    if [ -n "$num" ] && [ "$num" -gt "$MAX_NUM" ]; then
        MAX_NUM=$num
    fi
done

NEXT_NUM=$((MAX_NUM + 1))
echo "  Continuing from job_${NEXT_NUM}.txt"

# Split new entries into chunks
CHUNK_NUM=$((NEXT_NUM - 1))
LINE_NUM=0
CHUNKS_CREATED=0

while IFS= read -r line || [ -n "$line" ]; do
    if [ $((LINE_NUM % JOBS_PER_CHUNK)) -eq 0 ]; then
        CHUNK_NUM=$((CHUNK_NUM + 1))
        CHUNK_FILE="job_${CHUNK_NUM}.txt"
        > "$CHUNK_FILE"
        CHUNKS_CREATED=$((CHUNKS_CREATED + 1))
    fi
    echo "$line" >> "$CHUNK_FILE"
    LINE_NUM=$((LINE_NUM + 1))
done < "$NEW_LIST_FILE"

rm -f "$NEW_LIST_FILE"

echo "  Created $CHUNKS_CREATED new chunk(s): job_${NEXT_NUM}.txt → job_${CHUNK_NUM}.txt"
echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Restart submit monitor (health & backup stay alive)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5]${NC} Restarting submit monitor..."

# Check if submit session exists and whether it's still actively submitting
if tmux has-session -t submit 2>/dev/null; then
    echo "  Stopping existing submit session..."
    tmux kill-session -t submit 2>/dev/null || true
    sleep 2
fi

tmux new-session -d -s submit "cd $SCRIPT_DIR && ./monitor_submit.sh"
echo -e "  Started: ${GREEN}submit${NC} (will pick up new job_*.txt files)"

# Confirm health and backup are still running
echo ""
echo "  Other monitors:"
if tmux has-session -t health 2>/dev/null; then
    echo -e "    health: ${GREEN}running${NC}"
else
    echo -e "    health: ${RED}not running${NC} — consider: tmux new-session -d -s health './monitor_health.sh'"
fi
if tmux has-session -t backup 2>/dev/null; then
    echo -e "    backup: ${GREEN}running${NC}"
else
    echo -e "    backup: ${RED}not running${NC} — consider: tmux new-session -d -s backup './backup_results.sh'"
fi

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                   ✅ New Data Added!                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  New .zip files:      $NEW_COUNT"
echo "  New chunks created:  $CHUNKS_CREATED (× $JOBS_PER_CHUNK)"
echo "  Total tracked jobs:  $TOTAL_COUNT"
echo ""
echo "  The submit monitor will automatically pick up the new chunks."
echo "  Health and backup monitors were NOT restarted."
echo ""
echo -e "${YELLOW}Tip:${NC} Move processed tar.gz files to a 'processed/' folder"
echo "  to avoid re-extracting them next time:"
echo "  mkdir -p processed && mv *.tar.gz processed/"
echo ""
