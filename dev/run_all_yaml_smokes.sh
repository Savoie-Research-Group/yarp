#!/usr/bin/env bash
set -u

log_dir="./logs"
aggregate_log="${log_dir}/all_yaml_runs.log"

mkdir -p "${log_dir}"
: > "${aggregate_log}"

yamls=(
  "./main_inputs/yarp_egat_smoke.yaml"
  "./main_inputs/yarp_egat_smoke_larger.yaml"
  "./main_inputs/yarp_egat_smoke_aromatic.yaml"
  "./main_inputs/yarp_egat_smoke_isotope.yaml"
  "./main_inputs/yarp_egat_smoke_sparse_map.yaml"
  "./main_inputs/yarp_failure_sparse_map_egat.yaml"
  "./main_inputs/yarp_failure_duplicate_map.yaml"
  "./main_inputs/yarp_egat_smoke_depth2.yaml"
)

echo "Running ${#yamls[@]} YAML smoke cases" | tee -a "${aggregate_log}"
echo "Working directory: $(pwd)" | tee -a "${aggregate_log}"

for yaml in "${yamls[@]}"; do
  name="$(basename "${yaml}" .yaml)"
  run_log="${log_dir}/${name}.log"
  echo "" | tee -a "${aggregate_log}"
  echo "===== ${name} =====" | tee -a "${aggregate_log}"
  echo "Command: python ../yarp/main_yarp.py ${yaml}" | tee -a "${aggregate_log}"
  if python ../yarp/main_yarp.py "${yaml}" > "${run_log}" 2>&1; then
    echo "STATUS: PASS" | tee -a "${aggregate_log}"
  else
    status=$?
    echo "STATUS: FAIL (${status})" | tee -a "${aggregate_log}"
  fi
  echo "LOG: ${run_log}" | tee -a "${aggregate_log}"
  tail -n 40 "${run_log}" | tee -a "${aggregate_log}"
done

echo "" | tee -a "${aggregate_log}"
echo "Aggregate log written to ${aggregate_log}" | tee -a "${aggregate_log}"
