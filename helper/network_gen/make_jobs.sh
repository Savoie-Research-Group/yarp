#!/bin/bash
# Build per-SMILES job directories + joblist from a CSV and copy a config into each job.
#
# Usage:
#   ./make_jobs.sh [csv] [output_base] [joblist] [smiles_col] [config_template]
#
# Defaults:
#   csv            = smiles.csv
#   output_base    = ./outputs
#   joblist        = <output_base>/joblist.txt
#   smiles_col     = 1
#   config_template= ./config.yaml

set -euo pipefail

CSV="${1:-missing_smiles.csv}"
OUTBASE="${2:-./outputs}"
JOBLIST="${3:-$OUTBASE/joblist.txt}"
SMI_COL="${4:-1}"
CONFIG_TEMPLATE="${5:-./config.yaml}"

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: CSV not found: $CSV" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_TEMPLATE" ]]; then
  echo "ERROR: Config template not found: $CONFIG_TEMPLATE" >&2
  exit 1
fi

mkdir -p "$OUTBASE"
mkdir -p "$(dirname "$JOBLIST")"
: > "$JOBLIST"

slugify() {
  echo "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'
}

hash8() {
  echo -n "$1" | md5sum | awk '{print substr($1,1,8)}'
}

trim() {
  echo "$1" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

write_job_config() {
  local src="$1"
  local dst="$2"
  awk '
    /^[[:space:]]*smiles_txt:[[:space:]]*/ { print "  smiles_txt: \"./smi.txt\""; next }
    /^[[:space:]]*work_dir:[[:space:]]*/   { print "  work_dir: \".\""; next }
    { print }
  ' "$src" > "$dst"
}

echo "Reading CSV:       $CSV"
echo "Output base:       $OUTBASE"
echo "Joblist:           $JOBLIST"
echo "SMILES column:     $SMI_COL"
echo "Config template:   $CONFIG_TEMPLATE"

awk -v col="$SMI_COL" -F',' '
function trim(s) { sub(/^[ \t\r\n]+/, "", s); sub(/[ \t\r\n]+$/, "", s); return s }
{
  if (NF < col) next
  s = $col
  s = trim(s)
  gsub(/^"/, "", s); gsub(/"$/, "", s)
  s = trim(s)
  if (s == "") next
  l = tolower(s)
  if (l == "smiles" || l == "smile") next
  if (!(s in seen)) { seen[s]=1; print s }
}
' "$CSV" | while IFS= read -r s; do
  s="$(trim "$s")"
  [[ -z "$s" ]] && continue

  slug="$(slugify "$s")"
  [[ -z "$slug" ]] && slug="smiles"
  h="$(hash8 "$s")"

  dir="$OUTBASE/${slug}_${h}"
  mkdir -p "$dir"
  printf "%s\n" "$s" > "$dir/smi.txt"
  write_job_config "$CONFIG_TEMPLATE" "$dir/config.yaml"
  printf "%s\n" "$dir" >> "$JOBLIST"
done

N="$(wc -l < "$JOBLIST" | tr -d '[:space:]')"
echo "Wrote $N jobs to $JOBLIST"

if [[ "${N:-0}" -eq 0 ]]; then
  echo "ERROR: No SMILES found. Check CSV path/column." >&2
  exit 1
fi
