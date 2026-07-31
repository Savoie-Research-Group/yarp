#!/bin/bash
# Export conda env from egat_container_env for reproducibility.
# Run: ./export_env.sh  (with egat_container_env active or as default)
set -e
cd "$(dirname "$0")"
TMP=$(mktemp)
if conda env export -n egat_container_env --no-builds > "$TMP" 2>/dev/null; then
    :
elif conda env export --no-builds > "$TMP" 2>/dev/null; then
    :
else
    echo "ERROR: conda env export failed; not writing environment_exported.yml" >&2
    cat "$TMP" >&2
    rm -f "$TMP"
    exit 1
fi

# Sanity check: avoid writing a file that is actually an error message.
if ! grep -q '^name:' "$TMP"; then
    echo "ERROR: export output doesn't look like a conda environment YAML; not writing environment_exported.yml" >&2
    cat "$TMP" >&2
    rm -f "$TMP"
    exit 1
fi

# Remove prefix line so micromamba can use -p /opt/egat-env in container
grep -v '^prefix:' "$TMP" > environment_exported.yml
rm -f "$TMP"
echo "Saved to environment_exported.yml"
