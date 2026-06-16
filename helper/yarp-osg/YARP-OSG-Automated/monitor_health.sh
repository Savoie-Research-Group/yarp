#!/bin/bash
# DFT-style health monitor for YARP OSG jobs.

set -u

WORK_DIR="${1:-$PWD}"
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
STATE_DIR="$WORK_DIR/.yarp_osg"
CHECK_INTERVAL="${YARP_OSG_CHECK_INTERVAL:-300}"
USER_ID="${USER}"

RESOURCE_LEVELS=(
    "8000:8"
    "12000:8"
    "20000:10"
    "32000:12"
)
MAX_LEVEL=$(( ${#RESOURCE_LEVELS[@]} - 1 ))
DISK_UPGRADE=41943040

LOG_FILE="$STATE_DIR/health_monitor.log"
UNKNOWN_HOLDS_LOG="$STATE_DIR/held_jobs_unknown.log"
MAXED_JOBS_LOG="$STATE_DIR/held_jobs_maxed.log"
RESOURCE_TRACKING="$STATE_DIR/resource_tracking.dat"

mkdir -p "$STATE_DIR"
touch "$RESOURCE_TRACKING" "$MAXED_JOBS_LOG"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

level_mem() {
    echo "${RESOURCE_LEVELS[$1]}" | cut -d: -f1
}

level_cpus() {
    echo "${RESOURCE_LEVELS[$1]}" | cut -d: -f2
}

track_level() {
    job_id="$1"
    level="$2"
    grep -v "^${job_id} " "$RESOURCE_TRACKING" > "${RESOURCE_TRACKING}.tmp" 2>/dev/null || true
    mv "${RESOURCE_TRACKING}.tmp" "$RESOURCE_TRACKING" 2>/dev/null || true
    echo "$job_id $level $(date +%s)" >> "$RESOURCE_TRACKING"
}

tracked_level() {
    grep "^$1 " "$RESOURCE_TRACKING" 2>/dev/null | tail -1 | awk '{print $2}'
}

handle_transfer_holds() {
    condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -iE "transfer|OSDF|Pelican|\.sif|container" | while read -r cluster proc reason; do
            [ -z "$cluster" ] && continue
            job_id="${cluster}.${proc}"
            log "Releasing transfer/container-held job $job_id: $reason"
            condor_release "$job_id" 2>/dev/null || true
        done
}

handle_resource_holds() {
    condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId RequestMemory RequestCpus HoldReason 2>/dev/null | \
        grep -iE "memory|cpu|resource" | while read -r cluster proc mem cpus reason; do
            [ -z "$cluster" ] && continue
            job_id="${cluster}.${proc}"
            current="$(tracked_level "$job_id")"
            [ -z "$current" ] && current=0
            next=$((current + 1))
            if [ "$next" -gt "$MAX_LEVEL" ]; then
                log "Job $job_id maxed resources; not releasing"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $job_id $reason" >> "$MAXED_JOBS_LOG"
                continue
            fi
            new_mem="$(level_mem "$next")"
            new_cpus="$(level_cpus "$next")"
            log "Escalating $job_id to ${new_mem}MB/${new_cpus}CPU"
            condor_qedit "$job_id" RequestMemory "$new_mem" 2>/dev/null || true
            condor_qedit "$job_id" RequestCpus "$new_cpus" 2>/dev/null || true
            track_level "$job_id" "$next"
            condor_release "$job_id" 2>/dev/null || true
        done
}

handle_disk_holds() {
    condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -i "disk" | while read -r cluster proc reason; do
            [ -z "$cluster" ] && continue
            job_id="${cluster}.${proc}"
            log "Increasing disk for $job_id"
            condor_qedit "$job_id" RequestDisk "$DISK_UPGRADE" 2>/dev/null || true
            condor_release "$job_id" 2>/dev/null || true
        done
}

log_unknown_holds() {
    condor_q "$USER_ID" -constraint 'JobStatus == 5' -af ClusterId ProcId HoldReason 2>/dev/null | \
        grep -viE "transfer|OSDF|Pelican|\.sif|container|memory|cpu|resource|disk" >> "$UNKNOWN_HOLDS_LOG" || true
}

log "YARP OSG health monitor started for $WORK_DIR"
while true; do
    handle_transfer_holds
    handle_resource_holds
    handle_disk_holds
    log_unknown_holds
    sleep "$CHECK_INTERVAL"
done
