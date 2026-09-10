#!/bin/bash
# file: backup_results.sh
# Periodically packs old result files and moves them to long-term storage
#
# Usage:
#   ./backup_results.sh                # Run in foreground (or from tmux)
#   ./backup_results.sh --run-once     # Run one backup cycle and exit
#   ./backup_results.sh --status       # Check backup state
#   ./backup_results.sh --help         # Show help
#
# What it does:
#   - Every 4 hours, finds result files older than 4 hours in results/
#   - Waits for at least 100 files before packing
#   - Creates a verified tar.gz archive at the destination
#   - Only deletes originals after successful verification
#   - Alerts if destination storage > 90% full

# --- CONFIGURATION ---
CHECK_INTERVAL=14400        # 4 hours in seconds
MIN_FILE_AGE=14400          # 4 hours — files must be older than this
MIN_QUIET_TIME=600          # 10 minutes — file must not be modified recently
MIN_FILES_TO_PACK=100       # Wait for at least this many files
DISK_ALERT_THRESHOLD=90     # Alert if destination > 90% full

# Paths
RESULTS_DIR=""              # Auto-detected from ./results/
DEST_BASE="/ospool/uw-shared/projects/ND_Savoie/$USER"
PROJECT_NAME=""             # Auto-detected from parent directory

# Logging
LOG_FILE="backup_results.log"
MANIFEST_DIR=".backup_manifests"
# ---------------------

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}SUCCESS:${NC} $1" | tee -a "$LOG_FILE"
}

# Auto-detect paths
setup_paths() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Auto-detect results directory
    if [ -z "$RESULTS_DIR" ]; then
        if [ -d "$SCRIPT_DIR/results" ]; then
            RESULTS_DIR="$SCRIPT_DIR/results"
        else
            log_error "Cannot find results/ directory."
            exit 1
        fi
    fi

    # Auto-detect project name from parent directory
    if [ -z "$PROJECT_NAME" ]; then
        local workflow_parent
        workflow_parent=$(dirname "$SCRIPT_DIR")
        PROJECT_NAME=$(basename "$workflow_parent")

        if [ "$PROJECT_NAME" = "home" ] || [ "$PROJECT_NAME" = "$USER" ]; then
            PROJECT_NAME="dft_energy_results"
        fi
    fi

    # Set destination directory
    DEST_DIR="$DEST_BASE/dft_energy_backups/$PROJECT_NAME"

    # Create manifest directory
    mkdir -p "$MANIFEST_DIR"
}

# Check destination disk usage
check_disk_usage() {
    if [ ! -d "$DEST_BASE" ]; then
        log_warn "Destination base $DEST_BASE does not exist yet"
        return 0
    fi

    local usage
    usage=$(df "$DEST_BASE" 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')

    if [ -n "$usage" ] && [ "$usage" -ge "$DISK_ALERT_THRESHOLD" ]; then
        log_error "ALERT: Destination storage is ${usage}% full! (threshold: ${DISK_ALERT_THRESHOLD}%)"
        log_error "Location: $DEST_BASE"
        return 1
    elif [ -n "$usage" ]; then
        log "Destination disk usage: ${usage}%"
    fi

    return 0
}

# Find files eligible for backup (recursive search for nested structure)
find_eligible_files() {
    local now=$(date +%s)
    local min_age_cutoff=$((now - MIN_FILE_AGE))
    local quiet_cutoff=$((now - MIN_QUIET_TIME))

    find "$RESULTS_DIR" -name "*_results.tar.gz" -type f 2>/dev/null | while read -r file; do
        local mtime
        mtime=$(stat -c %Y "$file" 2>/dev/null)
        [ -z "$mtime" ] && continue

        # Check if old enough and quiet
        if [ "$mtime" -lt "$min_age_cutoff" ] && [ "$mtime" -lt "$quiet_cutoff" ]; then
            echo "$file"
        fi
    done
}

# Create backup archive
create_backup() {
    local file_list="$1"
    local file_count="$2"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="backup_${timestamp}_n${file_count}.tar.gz"
    local backup_tmp="$DEST_DIR/${backup_name}.tmp"
    local backup_final="$DEST_DIR/${backup_name}"
    local manifest_file="$MANIFEST_DIR/manifest_${timestamp}.txt"

    log "Starting backup: $backup_name ($file_count files)"

    # Create destination directory
    if ! mkdir -p "$DEST_DIR" 2>/dev/null; then
        log_error "Cannot create destination directory: $DEST_DIR"
        return 1
    fi

    # Check disk space
    if ! check_disk_usage; then
        log_error "Aborting backup due to disk space issues"
        return 1
    fi

    # Write manifest
    echo "# Backup manifest: $backup_name" > "$manifest_file"
    echo "# Created: $(date)" >> "$manifest_file"
    echo "# Status: IN_PROGRESS" >> "$manifest_file"
    echo "# Destination: $backup_final" >> "$manifest_file"
    echo "# Files:" >> "$manifest_file"
    echo "$file_list" >> "$manifest_file"

    # Create tar.gz preserving directory structure relative to results/
    log "Creating archive at $backup_tmp..."

    if ! echo "$file_list" | tar -czf "$backup_tmp" -C "$RESULTS_DIR" --files-from=<(
        echo "$file_list" | while IFS= read -r f; do
            # Convert absolute path to relative path from RESULTS_DIR
            echo "${f#$RESULTS_DIR/}"
        done
    ) 2>/dev/null; then
        log_error "Failed to create archive: $backup_tmp"
        echo "# Status: FAILED_CREATE" >> "$manifest_file"
        rm -f "$backup_tmp"
        return 1
    fi

    # Verify archive integrity
    log "Verifying archive integrity..."
    if ! tar -tzf "$backup_tmp" > /dev/null 2>&1; then
        log_error "Archive verification failed: $backup_tmp"
        echo "# Status: FAILED_VERIFY" >> "$manifest_file"
        rm -f "$backup_tmp"
        return 1
    fi

    local archive_size
    archive_size=$(du -h "$backup_tmp" 2>/dev/null | cut -f1)
    log "Archive verified: $archive_size"

    # Atomic rename
    if ! mv "$backup_tmp" "$backup_final"; then
        log_error "Failed to rename archive"
        echo "# Status: FAILED_RENAME" >> "$manifest_file"
        return 1
    fi

    log_success "Archive created: $backup_final ($archive_size)"

    # Now safe to delete originals
    log "Removing original files..."
    local deleted=0
    local failed=0

    while IFS= read -r file; do
        [ -z "$file" ] && continue
        if rm -f "$file" 2>/dev/null; then
            deleted=$((deleted + 1))
        else
            log_warn "Failed to delete: $file"
            failed=$((failed + 1))
        fi
    done <<< "$file_list"

    log "Deleted $deleted files ($failed failed)"

    # Clean up empty directories left behind
    find "$RESULTS_DIR" -type d -empty -delete 2>/dev/null || true

    # Update manifest
    sed -i 's/IN_PROGRESS/COMPLETE/' "$manifest_file"
    echo "# Completed: $(date)" >> "$manifest_file"
    echo "# Deleted: $deleted, Failed: $failed" >> "$manifest_file"

    log_success "Backup complete: $backup_name"
    return 0
}

# Check for and resume incomplete backups
resume_incomplete() {
    local incomplete
    incomplete=$(grep -l "IN_PROGRESS\|FAILED" "$MANIFEST_DIR"/manifest_*.txt 2>/dev/null || true)

    if [ -z "$incomplete" ]; then
        return 0
    fi

    log_warn "Found incomplete backup manifest(s), checking..."

    for manifest in $incomplete; do
        local dest=$(grep "^# Destination:" "$manifest" | cut -d' ' -f3)
        local tmp_file="${dest}.tmp"
        local status=$(grep "^# Status:" "$manifest" | tail -1 | cut -d' ' -f3)

        case "$status" in
            IN_PROGRESS)
                if [ -f "$tmp_file" ]; then
                    log "Found incomplete archive: $tmp_file"
                    if tar -tzf "$tmp_file" > /dev/null 2>&1; then
                        log "Archive is valid, completing..."
                        if mv "$tmp_file" "$dest"; then
                            grep -v "^#" "$manifest" | while read -r file; do
                                [ -f "$file" ] && rm -f "$file"
                            done
                            sed -i 's/IN_PROGRESS/COMPLETE/' "$manifest"
                            log_success "Resumed and completed: $dest"
                        fi
                    else
                        log_warn "Archive is corrupt, will retry"
                        rm -f "$tmp_file"
                    fi
                fi
                ;;
            FAILED_*)
                log_warn "Previous backup failed ($status), will retry"
                rm -f "$manifest"
                ;;
        esac
    done
}

# Main backup cycle
run_backup_cycle() {
    log "=============================================="
    log "Starting backup cycle"
    log "Results dir: $RESULTS_DIR"
    log "Destination: $DEST_DIR"
    log "=============================================="

    # Check for incomplete backups first
    resume_incomplete

    # Find eligible files
    log "Scanning for files older than $((MIN_FILE_AGE / 3600)) hours..."
    local eligible_files
    eligible_files=$(find_eligible_files)

    if [ -z "$eligible_files" ]; then
        log "No eligible files found"
        return 0
    fi

    local file_count
    file_count=$(echo "$eligible_files" | grep -c .)
    log "Found $file_count eligible files"

    # Check minimum threshold
    if [ "$file_count" -lt "$MIN_FILES_TO_PACK" ]; then
        log "Waiting for at least $MIN_FILES_TO_PACK files (currently $file_count)"
        return 0
    fi

    # Create backup
    create_backup "$eligible_files" "$file_count"
}

# Status banner
print_status_banner() {
    local results_count=$(find "$RESULTS_DIR" -name "*_results.tar.gz" -type f 2>/dev/null | wc -l)
    local eligible_count=$(find_eligible_files | wc -l)
    local backup_count=$(ls "$DEST_DIR"/*.tar.gz 2>/dev/null | wc -l)

    printf "\n"
    printf "╔════════════════════════════════════════════════════════════╗\n"
    printf "║          DFT Energy — Results Backup Monitor               ║\n"
    printf "╠════════════════════════════════════════════════════════════╣\n"
    printf "║  Results directory: %-37s  ║\n" "results/"
    printf "║  Files in results:  %6d                                  ║\n" "$results_count"
    printf "║  Eligible (>4hrs):  %6d                                  ║\n" "$eligible_count"
    printf "║  Min threshold:     %6d                                  ║\n" "$MIN_FILES_TO_PACK"
    printf "╠════════════════════════════════════════════════════════════╣\n"
    printf "║  Backups created:   %6d                                  ║\n" "$backup_count"
    printf "║  Destination: %-42s  ║\n" "$(basename "$DEST_DIR")"
    printf "╚════════════════════════════════════════════════════════════╝\n"
    printf "  Last check: %s\n" "$(date '+%H:%M:%S')"
    printf "  Next check in %d hours\n" "$((CHECK_INTERVAL / 3600))"
    printf "\n"
}

# Main function
main() {
    case "${1:-}" in
        --run-once)
            setup_paths
            run_backup_cycle
            exit 0
            ;;
        --status)
            setup_paths
            echo "Recent log entries:"
            tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log file yet)"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--run-once|--status|--help]"
            echo ""
            echo "Options:"
            echo "  --run-once  Run one backup cycle and exit"
            echo "  --status    Show recent log entries"
            echo "  --help      Show this help"
            echo ""
            echo "Configuration:"
            echo "  CHECK_INTERVAL=$CHECK_INTERVAL seconds ($((CHECK_INTERVAL/3600)) hours)"
            echo "  MIN_FILE_AGE=$MIN_FILE_AGE seconds ($((MIN_FILE_AGE/3600)) hours)"
            echo "  MIN_FILES_TO_PACK=$MIN_FILES_TO_PACK"
            echo "  DEST_BASE=$DEST_BASE"
            exit 0
            ;;
        "")
            # Running in foreground (or from tmux)
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac

    setup_paths

    log "=============================================="
    log "Backup monitor started"
    log "User: $USER"
    log "Results: $RESULTS_DIR"
    log "Destination: $DEST_DIR"
    log "Check interval: $((CHECK_INTERVAL / 3600)) hours"
    log "Min file age: $((MIN_FILE_AGE / 3600)) hours"
    log "Min files to pack: $MIN_FILES_TO_PACK"
    log "=============================================="

    print_status_banner

    # Run first cycle immediately
    run_backup_cycle
    print_status_banner

    # Main loop
    while true; do
        sleep "$CHECK_INTERVAL"
        run_backup_cycle
        print_status_banner
    done
}

main "$@"
