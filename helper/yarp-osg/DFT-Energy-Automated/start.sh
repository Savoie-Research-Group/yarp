#!/bin/bash
# file: start.sh
# One-button launcher for DFT Energy (ORCA + Multiwfn WBO) workflow on OSPool
#
# Usage:
#   ./start.sh                  # Default: extract, prepare, submit, monitor
#   ./start.sh --help           # Show help
#
# Requirements:
#   1. Place this script and all other .sh/.submit files in a working directory
#   2. Place your input *.tar.gz file(s) in the same directory
#   3. Run: ./start.sh
#
# The workflow:
#   1. Extracts input tar.gz → inputs/ (preserving nested subdirectory structure)
#   2. Generates file_list.txt mapping each .zip to its relative output path
#   3. Pre-creates mirrored directory structure under results/
#   4. Splits file_list.txt into 1000-job chunks for staged submission
#   5. Launches tmux monitors: submit (drip-feeds jobs), health (auto-heals),
#      backup (archives results to /ospool storage)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ======================== CONFIGURATION ========================
# Container image (each user must have their own copy at this path)
CONTAINER_IMAGE="osdf:///ospool/ap40/data/${USER}/orca_6_1_1-zip-multiwfn-2.sif"

# How many jobs per chunk file (fed to condor one chunk at a time)
JOBS_PER_CHUNK=1000

# Max concurrent jobs in the queue
MAX_JOBS=10000
# ===============================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "One-button launcher for DFT Energy (ORCA + Multiwfn WBO) workflow."
            echo ""
            echo "Options:"
            echo "  --help        Show this help"
            echo ""
            echo "Prerequisites:"
            echo "  1. Place all workflow scripts in one directory"
            echo "  2. Place your input *.tar.gz in the same directory"
            echo "  3. Run: ./start.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        DFT Energy (ORCA + WBO) Workflow — One Button        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Working directory: ${GREEN}$SCRIPT_DIR${NC}"
echo -e "  Container:         ${GREEN}$(basename "$CONTAINER_IMAGE")${NC}"
echo -e "  Jobs per chunk:    ${GREEN}$JOBS_PER_CHUNK${NC}"
echo -e "  Max queue size:    ${GREEN}$MAX_JOBS${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Preflight checks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/8]${NC} Checking prerequisites..."

REQUIRED_SCRIPTS="run_orca_wbo.sh monitor_submit.sh monitor_health.sh backup_results.sh batch_job.submit"
MISSING=""
for script in $REQUIRED_SCRIPTS; do
    if [ ! -f "$script" ]; then
        MISSING="$MISSING $script"
    fi
done

if [ -n "$MISSING" ]; then
    echo -e "${RED}Error: Missing required files:${NC}$MISSING"
    exit 1
fi

# Check for input tar.gz files (exclude any results packages)
INPUT_TARBALLS=$(ls *.tar.gz 2>/dev/null | grep -v 'results_package' | grep -v 'backup_' || true)
if [ -z "$INPUT_TARBALLS" ]; then
    echo -e "${RED}Error: No *.tar.gz input files found in $SCRIPT_DIR${NC}"
    echo ""
    echo "Please place your input tar.gz archive(s) in this directory."
    exit 1
fi

TARBALL_COUNT=$(echo "$INPUT_TARBALLS" | wc -w)
echo -e "  Found $TARBALL_COUNT input archive(s): $(echo $INPUT_TARBALLS | tr '\n' ' ')"
echo -e "  ${GREEN}✓ Prerequisites OK${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Make scripts executable
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/8]${NC} Making scripts executable..."
chmod +x *.sh 2>/dev/null || true
echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Create directories
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/8]${NC} Creating directories..."
mkdir -p inputs results logs submitted_batches
echo -e "  Created: inputs/ results/ logs/ submitted_batches/"
echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Extract input archives
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/8]${NC} Extracting input archives..."

for TARBALL in $INPUT_TARBALLS; do
    echo "  Extracting: $TARBALL"
    tar -xzf "$TARBALL" -C inputs/
done

# Count zip files
TOTAL_ZIPS=$(find inputs/ -name "*.zip" -type f | wc -l)
echo "  Total .zip files found: $TOTAL_ZIPS"

if [ "$TOTAL_ZIPS" -eq 0 ]; then
    echo -e "${RED}Error: No .zip files found after extraction.${NC}"
    echo "  Check that your tar.gz contains .zip input files."
    exit 1
fi

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Generate file_list.txt
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[5/8]${NC} Generating file list and mirroring directory structure..."

INPUTS_ABS="$(cd inputs && pwd)"

# Generate file_list.txt:
#   Column 1 (zip_abspath):  /absolute/path/to/inputs/subdir/05_0_1.zip
#   Column 2 (zip_relpath):  subdir/05_0_1  (relative to inputs/, no .zip)
> file_list.txt

find "$INPUTS_ABS" -name "*.zip" -type f | sort | while IFS= read -r zipfile; do
    # Relative path from inputs/ directory (e.g., SpringerNature/subdir/05_0_1.zip)
    relpath="${zipfile#$INPUTS_ABS/}"
    # Strip .zip extension for the output naming
    relpath_no_ext="${relpath%.zip}"
    echo "$zipfile, $relpath_no_ext" >> file_list.txt
done

TOTAL_JOBS=$(wc -l < file_list.txt)
echo "  Generated file_list.txt with $TOTAL_JOBS entries"

# Pre-create mirrored directory structure under results/
(cd inputs && find . -type d -exec mkdir -p "../results/{}" \;)
echo "  Mirrored directory structure under results/"

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 6: Configure submit file
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[6/8]${NC} Configuring submit file..."

RESULTS_DIR="$SCRIPT_DIR/results"

# Replace placeholders in submit file
sed -i "s|__CONTAINER_IMAGE__|$CONTAINER_IMAGE|g" batch_job.submit
sed -i "s|__RESULTS_DIR__|$RESULTS_DIR|g" batch_job.submit

echo "  Set container image: $(basename "$CONTAINER_IMAGE")"
echo "  Set results directory: $RESULTS_DIR"
echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 7: Split file_list.txt into chunks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[7/8]${NC} Splitting file list into ${JOBS_PER_CHUNK}-job chunks..."

# Remove any old job_*.txt files (but not submitted ones)
rm -f job_*.txt

# Split file_list.txt into numbered chunks
CHUNK_NUM=0
LINE_NUM=0

while IFS= read -r line || [ -n "$line" ]; do
    if [ $((LINE_NUM % JOBS_PER_CHUNK)) -eq 0 ]; then
        CHUNK_NUM=$((CHUNK_NUM + 1))
        CHUNK_FILE="job_${CHUNK_NUM}.txt"
        > "$CHUNK_FILE"
    fi
    echo "$line" >> "$CHUNK_FILE"
    LINE_NUM=$((LINE_NUM + 1))
done < file_list.txt

echo "  Created $CHUNK_NUM chunk file(s) from $TOTAL_JOBS jobs"
ls -la job_*.txt 2>/dev/null | awk '{print "    " $NF " (" $5 " bytes)"}'
echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 8: Start monitors
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[8/8]${NC} Starting background monitors..."

# Kill existing sessions if they exist
tmux kill-session -t submit 2>/dev/null || true
tmux kill-session -t health 2>/dev/null || true
tmux kill-session -t backup 2>/dev/null || true

# Start submission monitor
tmux new-session -d -s submit "cd $SCRIPT_DIR && ./monitor_submit.sh"
echo -e "  Started: ${GREEN}submit${NC} (job submission monitor)"

# Start health monitor
tmux new-session -d -s health "cd $SCRIPT_DIR && ./monitor_health.sh"
echo -e "  Started: ${GREEN}health${NC} (auto-heal monitor)"

# Start backup monitor
tmux new-session -d -s backup "cd $SCRIPT_DIR && ./backup_results.sh"
echo -e "  Started: ${GREEN}backup${NC} (results backup)"

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                  🚀 DFT Workflow Started!                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Total input .zip files: $TOTAL_ZIPS"
echo "  Total jobs:             $TOTAL_JOBS"
echo "  Job chunks:             $CHUNK_NUM (× $JOBS_PER_CHUNK)"
echo "  Results directory:      $RESULTS_DIR"
echo ""
echo -e "${YELLOW}Monitor your jobs:${NC}"
echo "  ./status.sh                  # Quick status check"
echo "  tmux attach -t submit        # Watch submission (Ctrl+B, D to detach)"
echo "  tmux attach -t health        # Watch health monitor"
echo "  tmux attach -t backup        # Watch backup monitor"
echo "  condor_q                     # HTCondor queue status"
echo ""
echo -e "${YELLOW}To stop everything:${NC}"
echo "  tmux kill-session -t submit"
echo "  tmux kill-session -t health"
echo "  tmux kill-session -t backup"
echo "  condor_rm \$USER              # Remove all queued jobs"
echo ""
