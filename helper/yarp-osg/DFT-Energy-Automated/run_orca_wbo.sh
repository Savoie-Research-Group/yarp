#!/bin/bash
# file: run_orca_wbo.sh
# Runs ORCA DFT single-point energy + Multiwfn bond order analysis inside container
#
# Called by HTCondor on the worker node. The zip file is already transferred locally.
#
# Input:  XX_charge_mult.zip containing .xyz geometry files
# Output: results.tar.gz containing .out, .xyz, .inp, bond order matrices, status.json
#
# Molecules processed (in order):
#   finished_last, finished_last_opt, ts_final_geometry,
#   finished_first, finished_first_opt, input
#
# Bond orders computed: Mayer, Wiberg (Lowdin), Fuzzy
#
# Features:
#   - Wall-time awareness: stops cleanly before the 20-hour cluster limit
#   - Background watchdog: kills long-running ORCA if wall time is nearly up
#   - skip.txt support: skips molecules already completed in a previous round
#   - status.json: records per-molecule completion status for round-2 resubmission

# ==========================================
# 1. Metadata & Setup
# ==========================================

ZIP_ARGUMENT=$1

if [ -z "$ZIP_ARGUMENT" ]; then
    echo "Error: No zip file provided."
    exit 1
fi

# Get the filename (e.g., "05_-1_1.zip")
ZIP_FILENAME=$(basename "$ZIP_ARGUMENT")
# Get basename without extension (e.g., "05_-1_1")
ZIP_BASENAME=$(basename "$ZIP_FILENAME" .zip)

# Parse Charge (2nd to last value) and Multiplicity (last value)
CHARGE=$(echo "$ZIP_BASENAME" | awk -F_ '{print $(NF-1)}')
MULT=$(echo "$ZIP_BASENAME" | awk -F_ '{print $NF}')

echo "Processing $ZIP_FILENAME"
echo "-> Extracted Metadata: Charge=$CHARGE, Multiplicity=$MULT"

# --- Wall-time configuration ---
WALL_SECONDS=${WALL_SECONDS:-70200}       # 19.5 hours default
BUFFER_SECONDS=${BUFFER_SECONDS:-7200}    # 2 hour buffer per molecule
WATCHDOG_BUFFER=1800                      # 30 min: watchdog kills ORCA at WALL - 30min
START_TIME=$(date +%s)

echo "-> Wall limit: $((WALL_SECONDS / 3600))h $((WALL_SECONDS % 3600 / 60))m"
echo "-> Buffer per molecule: $((BUFFER_SECONDS / 3600))h"

# Helper: seconds remaining
time_remaining() {
    local now=$(date +%s)
    local elapsed=$((now - START_TIME))
    echo $((WALL_SECONDS - elapsed))
}

# Helper: check if we have enough time for another molecule
have_time() {
    local remaining=$(time_remaining)
    if [ "$remaining" -gt "$BUFFER_SECONDS" ]; then
        return 0  # true: we have time
    else
        return 1  # false: not enough time
    fi
}

# Set up Environment
export PATH="/opt/orca:${PATH}"
export LD_LIBRARY_PATH="/opt/orca:${LD_LIBRARY_PATH}"

# Fix OpenMPI Slot Error (critical for multi-core jobs)
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_hwloc_base_binding_policy=none

# ==========================================
# 2. Watchdog Process
# ==========================================
# Fires at (WALL_SECONDS - WATCHDOG_BUFFER) to kill any running ORCA process.
# This ensures we have time to pack results before the cluster kills us.

WATCHDOG_DELAY=$((WALL_SECONDS - WATCHDOG_BUFFER))
WATCHDOG_PID=""

start_watchdog() {
    (
        sleep "$WATCHDOG_DELAY"
        echo ""
        echo "========================================"
        echo "WATCHDOG: Wall time limit approaching!"
        echo "WATCHDOG: Killing ORCA processes..."
        echo "========================================"
        # Kill orca processes belonging to this job
        pkill -f "/opt/orca/orca" 2>/dev/null || true
        # Also kill any MPI children
        pkill -f "orca_" 2>/dev/null || true
        sleep 5
        # Force kill if still alive
        pkill -9 -f "/opt/orca/orca" 2>/dev/null || true
        pkill -9 -f "orca_" 2>/dev/null || true
    ) &
    WATCHDOG_PID=$!
    echo "-> Watchdog armed: will fire in $((WATCHDOG_DELAY / 3600))h $((WATCHDOG_DELAY % 3600 / 60))m"
}

stop_watchdog() {
    if [ -n "$WATCHDOG_PID" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
        WATCHDOG_PID=""
    fi
}

# Clean up watchdog on exit
trap 'stop_watchdog' EXIT

start_watchdog

# ==========================================
# 3. Unzip and Detect Skip List
# ==========================================

# Unzip (flatten directories so XYZs are in current folder)
unzip -q -j "$ZIP_FILENAME"

# Check for skip.txt (from round-2 resubmission)
declare -A SKIP_MOLECULES
if [ -f "skip.txt" ]; then
    echo "-> Found skip.txt (round-2 mode)"
    while IFS= read -r mol_name || [ -n "$mol_name" ]; do
        mol_name=$(echo "$mol_name" | xargs)  # trim whitespace
        [ -z "$mol_name" ] && continue
        SKIP_MOLECULES["$mol_name"]=1
        echo "   Skipping (already complete): $mol_name"
    done < skip.txt
fi

# ==========================================
# 4. Execution Loop
# ==========================================

MOLECULES=("finished_last" "finished_last_opt" "ts_final_geometry" "finished_first" "finished_first_opt" "input")

TOTAL_MOLS=${#MOLECULES[@]}
MISSING_COUNT=0
PROCESSED_COUNT=0
SKIPPED_COUNT=0
TIMEOUT_COUNT=0
TIMED_OUT=0

# Per-molecule status tracking
declare -A MOL_STATUS

for MOL in "${MOLECULES[@]}"; do
    XYZ_FILE="${MOL}.xyz"

    # --- Check skip list (round-2: already completed) ---
    if [ -n "${SKIP_MOLECULES[$MOL]+x}" ]; then
        MOL_STATUS["$MOL"]="complete_previous_round"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        echo "-> $MOL: skipped (complete from previous round)"
        continue
    fi

    # --- Check wall time before starting this molecule ---
    if [ "$TIMED_OUT" -eq 1 ]; then
        MOL_STATUS["$MOL"]="timeout_skipped"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        continue
    fi

    if ! have_time; then
        REMAINING=$(time_remaining)
        echo ""
        echo "========================================"
        echo "TIMEOUT: Only $((REMAINING / 60)) minutes remaining."
        echo "TIMEOUT: Skipping $MOL and all remaining molecules."
        echo "========================================"
        MOL_STATUS["$MOL"]="timeout_skipped"
        TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
        TIMED_OUT=1
        continue
    fi

    if [ -f "$XYZ_FILE" ]; then
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
        echo "----------------------------------------"
        REMAINING=$(time_remaining)
        echo "Starting ${MOL}... ($((REMAINING / 60))m remaining)"

        # --- A. Generate ORCA Input ---
        cat > "${MOL}.inp" <<EOF
! wB97X-V def2-TZVP def2/J defgrid3 RIJCOSX CHELPG Loewdin Mayer
%scf
  MaxIter 500
end
%pal
  nproc 4
end
%maxcore 2500
%base "${MOL}"
*xyzfile ${CHARGE} ${MULT} ${XYZ_FILE}
EOF

        # --- B. Run ORCA ---
        echo "  -> Running ORCA..."
        MOL_START=$(date +%s)
        /opt/orca/orca "${MOL}.inp" > "${MOL}.out"
        ORCA_EXIT=$?
        MOL_END=$(date +%s)
        MOL_ELAPSED=$((MOL_END - MOL_START))
        echo "  -> ORCA finished (exit=$ORCA_EXIT, ${MOL_ELAPSED}s)"

        # Check if watchdog killed ORCA
        if [ $ORCA_EXIT -ne 0 ]; then
            REMAINING=$(time_remaining)
            if [ "$REMAINING" -lt "$WATCHDOG_BUFFER" ]; then
                echo "  -> ORCA was likely killed by watchdog (wall time)."
                MOL_STATUS["$MOL"]="timeout_killed"
                TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
                TIMED_OUT=1
                # Clean up partial files for this molecule
                rm -f "${MOL}.gbw" "${MOL}.out" "${MOL}.inp"
                rm -f *.gbw *.vpot *.dens *.tmp *.cis *.tx *.opt
                continue
            else
                echo "  -> ORCA failed (non-timeout)."
                MOL_STATUS["$MOL"]="orca_failed"
                rm -f *.gbw *.vpot *.dens *.tmp *.cis *.tx *.opt
                continue
            fi
        fi

        # --- C. Run Multiwfn Analysis ---
        if [ -f "${MOL}.gbw" ]; then
            echo "  -> Converting to Molden..."

            /opt/orca/orca_2mkl "${MOL}" -molden > /dev/null 2>&1

            if [ -f "${MOL}.molden.input" ]; then
                echo "  -> Running Multiwfn Bond Order Analyses..."

                # 1. Mayer Bond Order
                echo "     -> Calculating Mayer..."
                cat > run_mayer.txt <<EOF
9
1
y
0
q
EOF
                /opt/Multiwfn_bin/Multiwfn_noGUI "${MOL}.molden.input" < run_mayer.txt > "${MOL}_mayer_log.out" 2>/dev/null
                if [ -f bndmat.txt ]; then
                    mv bndmat.txt "${MOL}_mayer_mat.txt"
                fi

                # 2. Wiberg Bond Order (Lowdin)
                echo "     -> Calculating Wiberg..."
                cat > run_wiberg.txt <<EOF
9
3
y
0
q
EOF
                /opt/Multiwfn_bin/Multiwfn_noGUI "${MOL}.molden.input" < run_wiberg.txt > "${MOL}_wiberg_log.out" 2>/dev/null
                if [ -f bndmat.txt ]; then
                    mv bndmat.txt "${MOL}_wiberg_mat.txt"
                fi

                # 3. Fuzzy Bond Order
                echo "     -> Calculating Fuzzy..."
                cat > run_fuzzy.txt <<EOF
9
7
y
0
q
EOF
                /opt/Multiwfn_bin/Multiwfn_noGUI "${MOL}.molden.input" < run_fuzzy.txt > "${MOL}_fuzzy_log.out" 2>/dev/null
                if [ -f bndmat.txt ]; then
                    mv bndmat.txt "${MOL}_fuzzy_mat.txt"
                fi

                rm -f run_mayer.txt run_wiberg.txt run_fuzzy.txt
                MOL_STATUS["$MOL"]="complete"

            else
                echo "  -> Warning: Molden conversion failed."
                MOL_STATUS["$MOL"]="molden_failed"
            fi
        else
            echo "  -> Error: ORCA .gbw file not found. Calculation likely failed."
            MOL_STATUS["$MOL"]="orca_no_gbw"
        fi

        # --- D. Cleanup heavy files ---
        echo "  -> Cleaning up..."
        rm -f *.gbw *.vpot *.dens *.tmp *.cis *.tx *.opt *.molden.input

    else
        echo "Warning: $XYZ_FILE not found in zip archive, skipping."
        MOL_STATUS["$MOL"]="missing"
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

# Stop watchdog (we're done with computation)
stop_watchdog

# ==========================================
# 5. Generate status.json
# ==========================================

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

# Determine overall status
OVERALL_STATUS="complete"
if [ "$TIMEOUT_COUNT" -gt 0 ]; then
    OVERALL_STATUS="partial"
fi

# Count completed molecules (including from previous round)
COMPLETE_COUNT=0
for MOL in "${MOLECULES[@]}"; do
    s="${MOL_STATUS[$MOL]}"
    if [ "$s" = "complete" ] || [ "$s" = "complete_previous_round" ]; then
        COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
    fi
done

if [ "$COMPLETE_COUNT" -eq 0 ] && [ "$PROCESSED_COUNT" -eq 0 ]; then
    OVERALL_STATUS="all_molecules_missing"
fi

echo ""
echo "Generating status.json..."

cat > status.json <<STATUSEOF
{
  "zip_file": "$ZIP_FILENAME",
  "charge": "$CHARGE",
  "multiplicity": "$MULT",
  "status": "$OVERALL_STATUS",
  "wall_limit_seconds": $WALL_SECONDS,
  "elapsed_seconds": $TOTAL_ELAPSED,
  "total_molecules": $TOTAL_MOLS,
  "completed": $COMPLETE_COUNT,
  "timeout_skipped": $TIMEOUT_COUNT,
  "missing": $MISSING_COUNT,
  "molecules": {
STATUSEOF

FIRST=1
for MOL in "${MOLECULES[@]}"; do
    if [ $FIRST -eq 1 ]; then
        FIRST=0
    else
        echo "," >> status.json
    fi
    echo -n "    \"$MOL\": \"${MOL_STATUS[$MOL]:-unknown}\"" >> status.json
done

cat >> status.json <<STATUSEOF

  }
}
STATUSEOF

echo "-> Status: $OVERALL_STATUS ($COMPLETE_COUNT/$TOTAL_MOLS complete, ${TIMEOUT_COUNT} timed out, ${MISSING_COUNT} missing)"
echo "-> Elapsed: $((TOTAL_ELAPSED / 3600))h $((TOTAL_ELAPSED % 3600 / 60))m"

# ==========================================
# 6. Final Packing
# ==========================================
echo ""

if [ "$OVERALL_STATUS" = "all_molecules_missing" ] && [ "$PROCESSED_COUNT" -eq 0 ]; then
    echo "ERROR: No .xyz files found for any molecule."
    tar -czf results.tar.gz status.json
    echo "Created minimal results.tar.gz with status.json"
else
    echo "Packing results..."
    # Pack all useful output files + status.json
    find . -maxdepth 1 \( \
        -name "*.out" -o -name "*.xyz" -o -name "*.inp" -o \
        -name "*_mat.txt" -o -name "*_log.out" -o \
        -name "status.json" \
    \) -print0 | tar -czf results.tar.gz --null -T -

    echo "Packed results.tar.gz"
fi

echo ""
echo "=========================================="
echo "Done. Status: $OVERALL_STATUS"
echo "=========================================="
