#!/bin/bash
# file: prepare_round2.sh
# Prepares round-2 resubmission for jobs that timed out (status: "partial")
#
# Usage:
#   ./prepare_round2.sh                    # Scan local results/ and backups
#   ./prepare_round2.sh /path/to/results   # Scan specific directory
#
# What it does:
#   1. Scans result tarballs for status.json with "status": "partial"
#   2. For each partial job:
#      a. Extracts completed output files (.out, .inp, *_mat.txt, *_log.out)
#      b. Finds the original input .zip in inputs/
#      c. Creates a new round-2 .zip with:
#         - Original .xyz files for unfinished molecules
#         - Completed output files from round-1 (carried forward)
#         - skip.txt listing already-completed molecules
#   3. Generates file_list_round2.txt for submission
#
# After running, submit with:
#   ./add_data.sh  (if round2_inputs/ is placed under inputs/)
#   OR manually: use file_list_round2.txt with condor_submit
#
# The round-2 run_orca_wbo.sh will:
#   - Read skip.txt → skip completed molecules
#   - Compute only the remaining molecules
#   - Pack everything (old + new) into results.tar.gz
#   - The result overwrites/replaces the partial round-1 result

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           DFT Energy — Prepare Round-2 Resubmission         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ======================== CONFIGURATION ========================
ROUND2_DIR="round2_inputs"
WORK_TMP=".round2_tmp_$$"
# ===============================================================

# Determine where to scan for results
SCAN_DIRS=()

# Always check local results/
if [ -d "results" ]; then
    SCAN_DIRS+=("results")
fi

# Check backup location
DEST_BASE="/ospool/ap40/data/$USER"
PROJECT_PARENT=$(basename "$(dirname "$SCRIPT_DIR")")
BACKUP_DIR="$DEST_BASE/dft_energy_backups/$PROJECT_PARENT"
if [ -d "$BACKUP_DIR" ]; then
    SCAN_DIRS+=("$BACKUP_DIR")
fi

# Allow override from command line
if [ -n "$1" ] && [ -d "$1" ]; then
    SCAN_DIRS=("$1")
fi

if [ ${#SCAN_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No results directories found to scan.${NC}"
    exit 1
fi

echo "Scanning directories:"
for d in "${SCAN_DIRS[@]}"; do
    echo "  $d"
done
echo ""

# Create working directories
mkdir -p "$ROUND2_DIR"
mkdir -p "$WORK_TMP"
trap "rm -rf '$WORK_TMP'" EXIT

INPUTS_ABS=""
if [ -d "inputs" ]; then
    INPUTS_ABS="$(cd inputs && pwd)"
fi

# -----------------------------------------------------------------------------
# Step 1: Find all result tarballs with status=partial
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/3]${NC} Scanning for partial (timed-out) results..."

PARTIAL_LIST="$WORK_TMP/partial_jobs.txt"
> "$PARTIAL_LIST"

TOTAL_SCANNED=0
PARTIAL_COUNT=0

for SCAN_DIR in "${SCAN_DIRS[@]}"; do
    # Handle both direct result files and backup archives
    # First: direct result tarballs
    while IFS= read -r result_tarball; do
        [ -f "$result_tarball" ] || continue
        TOTAL_SCANNED=$((TOTAL_SCANNED + 1))

        # Try to extract just status.json
        EXTRACT_DIR="$WORK_TMP/check_$TOTAL_SCANNED"
        mkdir -p "$EXTRACT_DIR"

        if tar -xzf "$result_tarball" -C "$EXTRACT_DIR" --wildcards '*/status.json' 'status.json' 2>/dev/null || \
           tar -xzf "$result_tarball" -C "$EXTRACT_DIR" status.json 2>/dev/null; then

            STATUS_FILE=$(find "$EXTRACT_DIR" -name "status.json" -type f | head -1)

            if [ -n "$STATUS_FILE" ] && [ -f "$STATUS_FILE" ]; then
                STATUS=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATUS_FILE" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')

                if [ "$STATUS" = "partial" ]; then
                    PARTIAL_COUNT=$((PARTIAL_COUNT + 1))
                    # Record: result_tarball_path status_json_path
                    echo "$result_tarball $STATUS_FILE" >> "$PARTIAL_LIST"
                fi
            fi
        fi

        rm -rf "$EXTRACT_DIR"

        # Progress
        if [ $((TOTAL_SCANNED % 200)) -eq 0 ]; then
            echo "  Scanned $TOTAL_SCANNED results ($PARTIAL_COUNT partial so far)..."
        fi

    done < <(find "$SCAN_DIR" -name "*_results.tar.gz" -type f 2>/dev/null)

    # Second: check inside backup archives
    while IFS= read -r backup_archive; do
        [ -f "$backup_archive" ] || continue

        # List contents for result tarballs
        INNER_RESULTS=$(tar -tzf "$backup_archive" 2>/dev/null | grep "_results\.tar\.gz$" || true)
        [ -z "$INNER_RESULTS" ] && continue

        # Extract the backup to check individual results
        BACKUP_EXTRACT="$WORK_TMP/backup_extract_$$"
        mkdir -p "$BACKUP_EXTRACT"
        tar -xzf "$backup_archive" -C "$BACKUP_EXTRACT" 2>/dev/null || continue

        while IFS= read -r inner_result; do
            [ -z "$inner_result" ] && continue
            INNER_PATH="$BACKUP_EXTRACT/$inner_result"
            [ -f "$INNER_PATH" ] || continue

            TOTAL_SCANNED=$((TOTAL_SCANNED + 1))

            EXTRACT_DIR="$WORK_TMP/check_inner_$TOTAL_SCANNED"
            mkdir -p "$EXTRACT_DIR"

            if tar -xzf "$INNER_PATH" -C "$EXTRACT_DIR" --wildcards '*/status.json' 'status.json' 2>/dev/null || \
               tar -xzf "$INNER_PATH" -C "$EXTRACT_DIR" status.json 2>/dev/null; then

                STATUS_FILE=$(find "$EXTRACT_DIR" -name "status.json" -type f | head -1)

                if [ -n "$STATUS_FILE" ] && [ -f "$STATUS_FILE" ]; then
                    STATUS=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATUS_FILE" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')

                    if [ "$STATUS" = "partial" ]; then
                        PARTIAL_COUNT=$((PARTIAL_COUNT + 1))
                        echo "$INNER_PATH $STATUS_FILE" >> "$PARTIAL_LIST"
                    fi
                fi
            fi

            rm -rf "$EXTRACT_DIR"

        done <<< "$INNER_RESULTS"

        rm -rf "$BACKUP_EXTRACT"

    done < <(find "$SCAN_DIR" -name "backup_*.tar.gz" -type f 2>/dev/null)
done

echo "  Scanned: $TOTAL_SCANNED result tarballs"
echo -e "  Partial: ${GREEN}$PARTIAL_COUNT${NC}"
echo ""

if [ "$PARTIAL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}No partial results found — nothing to resubmit!${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 2: Build round-2 input zips
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/3]${NC} Building round-2 input zips..."

ROUND2_COUNT=0
SKIPPED_NO_ORIGINAL=0

> "$WORK_TMP/file_list_round2.txt"

while IFS=' ' read -r result_tarball status_file_ignored; do
    [ -z "$result_tarball" ] && continue
    [ -f "$result_tarball" ] || continue

    # Extract full result tarball to work area
    R2_WORK="$WORK_TMP/r2_build_$ROUND2_COUNT"
    mkdir -p "$R2_WORK/result_contents"

    tar -xzf "$result_tarball" -C "$R2_WORK/result_contents" 2>/dev/null || continue

    # Find status.json in extracted contents
    STATUS_FILE=$(find "$R2_WORK/result_contents" -name "status.json" -type f | head -1)
    [ -f "$STATUS_FILE" ] || { rm -rf "$R2_WORK"; continue; }

    # Parse zip_file name from status.json
    ZIP_NAME=$(grep -o '"zip_file"[[:space:]]*:[[:space:]]*"[^"]*"' "$STATUS_FILE" | sed 's/.*"\([^"]*\)"$/\1/')
    [ -z "$ZIP_NAME" ] && { rm -rf "$R2_WORK"; continue; }

    ZIP_BASENAME=$(basename "$ZIP_NAME" .zip)

    # Find completed molecules
    COMPLETED_MOLS=()
    INCOMPLETE_MOLS=()

    for mol in "finished_last" "finished_last_opt" "ts_final_geometry" "finished_first" "finished_first_opt" "input"; do
        mol_status=$(grep -o "\"$mol\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$STATUS_FILE" | sed 's/.*"\([^"]*\)"$/\1/')

        if [ "$mol_status" = "complete" ] || [ "$mol_status" = "complete_previous_round" ]; then
            COMPLETED_MOLS+=("$mol")
        elif [ "$mol_status" = "timeout_skipped" ] || [ "$mol_status" = "timeout_killed" ]; then
            INCOMPLETE_MOLS+=("$mol")
        fi
        # missing / orca_failed / etc. are not retried
    done

    if [ ${#INCOMPLETE_MOLS[@]} -eq 0 ]; then
        # Nothing to retry
        rm -rf "$R2_WORK"
        continue
    fi

    # Find original input zip
    ORIGINAL_ZIP=""
    if [ -n "$INPUTS_ABS" ]; then
        ORIGINAL_ZIP=$(find "$INPUTS_ABS" -name "$ZIP_NAME" -type f | head -1)
    fi

    if [ -z "$ORIGINAL_ZIP" ] || [ ! -f "$ORIGINAL_ZIP" ]; then
        echo "  Warning: Original zip not found for $ZIP_NAME — skipping"
        SKIPPED_NO_ORIGINAL=$((SKIPPED_NO_ORIGINAL + 1))
        rm -rf "$R2_WORK"
        continue
    fi

    # Build round-2 zip
    R2_BUILD="$R2_WORK/build"
    mkdir -p "$R2_BUILD"

    # A. Extract original zip to get .xyz files
    unzip -q -j "$ORIGINAL_ZIP" -d "$R2_BUILD"

    # B. Copy completed output files from round-1 result
    # These get carried forward so round-2 results.tar.gz is self-contained
    for completed_mol in "${COMPLETED_MOLS[@]}"; do
        for pattern in "${completed_mol}.out" "${completed_mol}.inp" \
                       "${completed_mol}_mayer_mat.txt" "${completed_mol}_wiberg_mat.txt" \
                       "${completed_mol}_fuzzy_mat.txt" "${completed_mol}_mayer_log.out" \
                       "${completed_mol}_wiberg_log.out" "${completed_mol}_fuzzy_log.out"; do
            found_file=$(find "$R2_WORK/result_contents" -name "$pattern" -type f | head -1)
            if [ -n "$found_file" ] && [ -f "$found_file" ]; then
                cp "$found_file" "$R2_BUILD/"
            fi
        done
    done

    # C. Create skip.txt
    > "$R2_BUILD/skip.txt"
    for completed_mol in "${COMPLETED_MOLS[@]}"; do
        echo "$completed_mol" >> "$R2_BUILD/skip.txt"
    done

    # D. Create the round-2 zip
    # Determine output path preserving original directory structure
    ORIGINAL_RELPATH="${ORIGINAL_ZIP#$INPUTS_ABS/}"
    ORIGINAL_RELDIR=$(dirname "$ORIGINAL_RELPATH")

    R2_DEST_DIR="$ROUND2_DIR/$ORIGINAL_RELDIR"
    mkdir -p "$R2_DEST_DIR"

    R2_ZIP="$R2_DEST_DIR/$ZIP_NAME"
    (cd "$R2_BUILD" && zip -q "$SCRIPT_DIR/$R2_ZIP" *)

    ROUND2_COUNT=$((ROUND2_COUNT + 1))

    # E. Add to file_list
    R2_ZIP_ABS="$(cd "$SCRIPT_DIR" && pwd)/$R2_ZIP"
    R2_RELPATH="${R2_ZIP#$ROUND2_DIR/}"
    R2_RELPATH_NO_EXT="${R2_RELPATH%.zip}"
    echo "$R2_ZIP_ABS, $R2_RELPATH_NO_EXT" >> "$WORK_TMP/file_list_round2.txt"

    echo "  [$ROUND2_COUNT] $ZIP_BASENAME: ${#COMPLETED_MOLS[@]} complete, ${#INCOMPLETE_MOLS[@]} to retry"

    rm -rf "$R2_WORK"

done < "$PARTIAL_LIST"

echo ""
echo "  Round-2 zips created: $ROUND2_COUNT"
if [ "$SKIPPED_NO_ORIGINAL" -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped (original zip not found): $SKIPPED_NO_ORIGINAL${NC}"
fi
echo ""

if [ "$ROUND2_COUNT" -eq 0 ]; then
    echo -e "${GREEN}No round-2 jobs to create.${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 3: Generate submission files
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/3]${NC} Generating submission files..."

# Copy file_list_round2.txt to script dir
cp "$WORK_TMP/file_list_round2.txt" "$SCRIPT_DIR/file_list_round2.txt"
echo "  Created: file_list_round2.txt ($ROUND2_COUNT entries)"

# Pre-create results directories for round-2 outputs
if [ -d "results" ]; then
    ROUND2_ABS="$(cd "$ROUND2_DIR" && pwd)"
    (cd "$ROUND2_DIR" && find . -type d -exec mkdir -p "$SCRIPT_DIR/results/{}" \;)
    echo "  Mirrored directory structure under results/"
fi

echo -e "  ${GREEN}✓ Done${NC}"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                 Round-2 Preparation Complete                ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Partial results found:  $PARTIAL_COUNT"
echo "  Round-2 zips created:   $ROUND2_COUNT"
echo "  Round-2 inputs:         $ROUND2_DIR/"
echo "  File list:              file_list_round2.txt"
echo ""
echo -e "${YELLOW}To submit round-2 jobs:${NC}"
echo ""
echo "  Option A — Move round-2 inputs into inputs/ and use add_data.sh:"
echo "    cp -r $ROUND2_DIR/* inputs/"
echo "    ./add_data.sh"
echo ""
echo "  Option B — Submit directly with condor_submit:"
echo "    condor_submit batch_job.submit input_list=file_list_round2.txt"
echo ""
echo "  Option C — Split and submit in chunks (recommended for many jobs):"
echo "    split -l 1000 -d -a 3 file_list_round2.txt job_r2_"
echo "    # Rename: for f in job_r2_*; do mv \"\$f\" \"\${f}.txt\"; done"
echo "    # Then restart submit monitor"
echo ""
