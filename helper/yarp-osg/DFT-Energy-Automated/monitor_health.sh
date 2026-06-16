#!/bin/bash
# file: monitor_health.sh
# Monitors job health and auto-heals common issues for DFT Energy jobs
#
# Usage:
#   ./monitor_health.sh
#   tmux new-session -d -s health './monitor_health.sh'
#
# Handles:
#   - SIF/transfer failures → auto-release
#   - Memory/CPU limit exceeded → escalate resources + release
#   - Disk limit exceeded → increase disk + release
#   - Unknown holds → log for manual review
#
# Resource Escalation (4 levels):
#   Level 0: 12 GB / 4 CPUs  (default)
#   Level 1: 20 GB / 6 CPUs
#   Level 2: 32 GB / 8 CPUs
#   Level 3: 40 GB / 10 CPUs  (MAX — not auto-released beyond this)
#
# Logs:
#   health_monitor.log      - All actions taken
#   held_jobs_unknown.log   - Jobs held for unknown reasons
#   held_jobs_maxed.log     - Jobs that hit max resources (need manual review)
#   resource_tracking.dat   - Tracks resource level per job

# --- CONFIGURATION ---
CHECK_INTERVAL=300          # Check every 5 minutes
USER_ID="${USER}"

# Combined resource levels: "MEMORY_MB:CPUS"
# Any resource failure (memory OR CPU) bumps the job to the next level
RESOURCE_LEVELS=(
    "12288:4"    # Level 0: 12 GB, 4 CPUs  (default)
    "20480:6"    # Level 1: 20 GB, 6 CPUs
    "32768:8"    # Level 2: 32 GB, 8 CPUs
    "40960:10"   # Level 3: 40 GB, 10 CPUs (MAX)
)

MAX_LEVEL=$(( ${#RESOURCE_LEVELS[@]} - 1 ))

# Disk upgrade (40 GB in KB)
DISK_UPGRADE=41943040

LOG_FILE="health_monitor.log"
UNKNOWN_HOLDS_LOG="held_jobs_unknown.log"
MAXED_JOBS_LOG="held_jobs_maxed.log"
RESOURCE_TRACKING="resource_tracking.dat"
# ---------------------

# Initialize tracking files
touch "$RESOURCE_TRACKING"
touch "$MAXED_JOBS_LOG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_unknown() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$UNKNOWN_HOLDS_LOG"
}

log_maxed() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$MAXED_JOBS_LOG"
}

get_job_counts() {
    # Returns: total idle running held completed
    condor_q "$USER_ID" -af JobStatus 2>/dev/null | \
        awk 'BEGIN {t=0;i=0;r=0;h=0;c=0}
             {t++; if($1==1)i++; if($1==2)r++; if($1==5)h++; if($1==4)c++}
             END {print t,i,r,h,c}'
}

# Get memory from a level index
get_level_memory() {
    echo "${RESOURCE_LEVELS[$1]}" | cut -d: -f1
}

# Get CPUs from a level index
get_level_cpus() {
    echo "${RESOURCE_LEVELS[$1]}" | cut -d: -f2
}

# Update resource tracking for a job
track_resource_level() {
    local job_id=$1
    local level_idx=$2

    # Remove old entry if exists
    grep -v "^${job_id} " "$RESOURCE_TRACKING" > "${RESOURCE_TRACKING}.tmp" 2>/dev/null || true
    mv "${RESOURCE_TRACKING}.tmp" "$RESOURCE_TRACKING" 2>/dev/null || true

    echo "${job_id} ${level_idx} $(date +%s)" >> "$RESOURCE_TRACKING"
}

# Get tracked resource level for a job
get_tracked_level() {
    local job_id=$1
    grep "^${job_id} " "$RESOURCE_TRACKING" 2>/dev/null | tail -1 | awk '{print $2}'
}

# Find the current level index based on memory value
find_level_by_memory() {
    local current_mem=$1

    for i in "${!RESOURCE_LEVELS[@]}"; do
        local level_mem=$(get_level_memory "$i")
        if [ "$level_mem" -eq "$current_mem" ] 2>/dev/null; then
            echo "$i"
            return
        fi
    done

    # If not found, find closest level that's >= current_mem
    for i in "${!RESOURCE_LEVELS[@]}"; do
        local level_mem=$(get_level_memory "$i")
        if [ "$level_mem" -ge "$current_mem" ] 2>/dev/null; then
            echo "$i"
            return
        fi
    done

    # Default to max
    echo "$MAX_LEVEL"
}

handle_transfer_failures() {
    local held_jobs
    held_jobs=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -iE "transfer|OSDF|\.sif" || true)

    if [ -z "$held_jobs" ]; then
        return 0
    fi

    local count=0
    while read -r cluster proc hold_reason; do
        [ -z "$cluster" ] && continue
        local job_id="${cluster}.${proc}"
        log "  Releasing transfer-held job: $job_id"
        condor_release "$job_id" 2>/dev/null
        count=$((count + 1))
    done <<< "$held_jobs"

    if [ "$count" -gt 0 ]; then
        log "Released $count transfer-held jobs"
    fi

    return $count
}

handle_resource_failures() {
    local held_jobs
    held_jobs=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId RequestMemory RequestCpus HoldReason 2>/dev/null | \
        grep -iE "memory|cpu" || true)

    if [ -z "$held_jobs" ]; then
        return 0
    fi

    local count=0
    local maxed_count=0
    local jobs_to_release=""

    while read -r cluster proc current_mem current_cpus hold_reason; do
        [ -z "$cluster" ] && continue

        local job_id="${cluster}.${proc}"

        # --- Determine current level ---

        local current_level=""

        # Tier 1: From condor's RequestMemory
        if [ -n "$current_mem" ] && [ "$current_mem" -gt 0 ] 2>/dev/null; then
            current_level=$(find_level_by_memory "$current_mem")
        fi

        # Tier 2: From tracking file
        if [ -z "$current_level" ]; then
            current_level=$(get_tracked_level "$job_id")
            if [ -n "$current_level" ]; then
                log "  Job $job_id: Using tracked level $current_level"
            fi
        fi

        # Tier 3: Conservative default (second-highest level)
        if [ -z "$current_level" ]; then
            current_level=$((MAX_LEVEL - 1))
            log "  WARNING: Job $job_id — cannot determine level, assuming level $current_level"
        fi

        # Calculate next level
        local next_level=$((current_level + 1))

        # Check if already at max
        if [ "$current_level" -ge "$MAX_LEVEL" ]; then
            local max_mem=$(get_level_memory "$MAX_LEVEL")
            local max_cpus=$(get_level_cpus "$MAX_LEVEL")
            log "  Job $job_id: MAXED OUT at level $MAX_LEVEL (${max_mem}MB, ${max_cpus} CPUs) — NOT releasing"
            log_maxed "Job $job_id: At max resources ($((max_mem/1024))GB, ${max_cpus} CPUs) — needs manual intervention"
            maxed_count=$((maxed_count + 1))
            continue
        fi

        # Get new resource values
        local new_mem=$(get_level_memory "$next_level")
        local new_cpus=$(get_level_cpus "$next_level")
        local old_mem=$(get_level_memory "$current_level")
        local old_cpus=$(get_level_cpus "$current_level")

        log "  Job $job_id: Level $current_level → $next_level ($((old_mem/1024))GB/${old_cpus}CPU → $((new_mem/1024))GB/${new_cpus}CPU)"

        # Apply new resources
        condor_qedit "$job_id" RequestMemory "$new_mem" 2>/dev/null
        condor_qedit "$job_id" RequestCpus "$new_cpus" 2>/dev/null

        # Track the new level
        track_resource_level "$job_id" "$next_level"

        # Add to release list
        jobs_to_release="$jobs_to_release $job_id"
        count=$((count + 1))

    done <<< "$held_jobs"

    # Release upgraded jobs (not maxed ones)
    if [ "$count" -gt 0 ]; then
        log "Upgraded $count jobs — releasing"
        sleep 2
        for job_id in $jobs_to_release; do
            condor_release "$job_id" 2>/dev/null
        done
    fi

    if [ "$maxed_count" -gt 0 ]; then
        log "WARNING: $maxed_count jobs at max resources — see $MAXED_JOBS_LOG"
    fi

    return $count
}

handle_disk_failures() {
    local held_jobs
    held_jobs=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -i "disk" || true)

    if [ -z "$held_jobs" ]; then
        return 0
    fi

    local count=0
    while read -r cluster proc hold_reason; do
        [ -z "$cluster" ] && continue
        local job_id="${cluster}.${proc}"
        log "  Job $job_id: Upgrading disk to $((DISK_UPGRADE / 1024 / 1024))GB"
        condor_qedit "$job_id" RequestDisk "$DISK_UPGRADE" 2>/dev/null
        condor_release "$job_id" 2>/dev/null
        count=$((count + 1))
    done <<< "$held_jobs"

    if [ "$count" -gt 0 ]; then
        log "Upgraded and released $count disk-held jobs"
    fi

    return $count
}

log_unknown_holds() {
    local held_jobs
    held_jobs=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -viE "transfer|OSDF|\.sif|memory|cpu|disk" || true)

    if [ -n "$held_jobs" ]; then
        local count
        count=$(echo "$held_jobs" | grep -c . || echo 0)
        if [ "$count" -gt 0 ]; then
            log "Warning: $count jobs still held for unhandled reasons (see $UNKNOWN_HOLDS_LOG)"
            while read -r cluster proc hold_reason; do
                [ -z "$cluster" ] && continue
                log_unknown "${cluster}.${proc}: $hold_reason"
            done <<< "$held_jobs"
        fi
    fi
}

cleanup_tracking_files() {
    if [ ! -f "$RESOURCE_TRACKING" ] || [ ! -s "$RESOURCE_TRACKING" ]; then
        return
    fi

    local current_jobs
    current_jobs=$(condor_q "$USER_ID" -af ClusterId ProcId 2>/dev/null | \
        awk '{print $1"."$2}' | sort -u)

    if [ -z "$current_jobs" ]; then
        > "$RESOURCE_TRACKING"
        return
    fi

    local current_jobs_pattern
    current_jobs_pattern=$(echo "$current_jobs" | tr '\n' '|' | sed 's/|$//')

    if [ -n "$current_jobs_pattern" ]; then
        grep -E "^($current_jobs_pattern) " "$RESOURCE_TRACKING" > "${RESOURCE_TRACKING}.tmp" 2>/dev/null || true
        mv "${RESOURCE_TRACKING}.tmp" "$RESOURCE_TRACKING" 2>/dev/null || true
    fi
}

count_maxed_jobs() {
    local count=0

    local held_jobs
    held_jobs=$(condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -iE "memory|cpu" || true)

    [ -z "$held_jobs" ] && echo 0 && return

    while read -r cluster proc hold_reason; do
        [ -z "$cluster" ] && continue
        local job_id="${cluster}.${proc}"
        local level=$(get_tracked_level "$job_id")
        if [ -n "$level" ] && [ "$level" -ge "$MAX_LEVEL" ]; then
            count=$((count + 1))
        fi
    done <<< "$held_jobs"

    echo "$count"
}

print_status() {
    read -r total idle running held completed <<< "$(get_job_counts)"

    local rate=0
    if [ "$total" -gt 0 ]; then
        rate=$((100 * completed / total))
    fi

    local tracked_jobs=$(wc -l < "$RESOURCE_TRACKING" 2>/dev/null || echo 0)
    local maxed_jobs=$(count_maxed_jobs)

    printf "\n"
    printf "╔════════════════════════════════════════════════════════════╗\n"
    printf "║          DFT Energy — Job Health Monitor                   ║\n"
    printf "╠════════════════════════════════════════════════════════════╣\n"
    printf "║  Total:     %6d                                         ║\n" "$total"
    printf "║  Idle:      %6d  ░░░░░░░░░░  Waiting                     ║\n" "$idle"
    printf "║  Running:   %6d  ██████████  Active                      ║\n" "$running"
    printf "║  Held:      %6d  ⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠  Need attention            ║\n" "$held"
    printf "║  Completed: %6d  (%3d%%)                                  ║\n" "$completed" "$rate"
    printf "╠════════════════════════════════════════════════════════════╣\n"
    printf "║  Jobs with resource upgrades: %5d                        ║\n" "$tracked_jobs"
    printf "║  Jobs at MAX (need review):   %5d                        ║\n" "$maxed_jobs"
    printf "╚════════════════════════════════════════════════════════════╝\n"
    printf "  Last check: %s\n" "$(date '+%H:%M:%S')"
    printf "  Next check in %d seconds\n" "$CHECK_INTERVAL"
    printf "\n"
    printf "  Resource levels (Memory : CPUs):\n"
    printf "    L0: 12GB:4  →  L1: 20GB:6  →  L2: 32GB:8  →  L3: 40GB:10 (MAX)\n"
}

# --- MAIN LOOP ---

log "=============================================="
log "Health monitor started"
log "User: $USER_ID"
log "Check interval: ${CHECK_INTERVAL}s"
log "Resource levels: ${RESOURCE_LEVELS[*]}"
log "Max level: $MAX_LEVEL"
log "=============================================="

# Initial status
print_status

CLEANUP_COUNTER=0

while true; do
    sleep "$CHECK_INTERVAL"

    # Check for held jobs and handle them
    transfer_fixed=0
    resource_fixed=0
    disk_fixed=0

    handle_transfer_failures
    transfer_fixed=$?

    handle_resource_failures
    resource_fixed=$?

    handle_disk_failures
    disk_fixed=$?

    total_fixed=$((transfer_fixed + resource_fixed + disk_fixed))

    if [ "$total_fixed" -gt 0 ]; then
        log "Auto-healed $total_fixed jobs (transfer:$transfer_fixed resource:$resource_fixed disk:$disk_fixed)"
    fi

    # Log any remaining held jobs
    log_unknown_holds

    # Periodic cleanup of tracking files (every 12 checks ≈ 1 hour at 5min intervals)
    CLEANUP_COUNTER=$((CLEANUP_COUNTER + 1))
    if [ $((CLEANUP_COUNTER % 12)) -eq 0 ]; then
        cleanup_tracking_files
    fi

    # Print status
    print_status

    # Check if we're done (no jobs left)
    read -r total idle running held completed <<< "$(get_job_counts)"
    if [ "$total" -eq 0 ]; then
        log "No jobs in queue — monitor complete"
        echo ""
        echo "All jobs processed! Check results with ./status.sh"
        break
    fi
done

log "Health monitor stopped"
