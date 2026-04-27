#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plot_script="$script_dir/plot_mocap_log.py"
output_dir="${1:-$script_dir/output}"

if [[ ! -f "$plot_script" ]]; then
    echo "Missing plot script: $plot_script" >&2
    exit 1
fi

mkdir -p "$output_dir"

shopt -s nullglob
csv_files=("$script_dir"/*.csv)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files found in $script_dir" >&2
    exit 1
fi

for csv_file in "${csv_files[@]}"; do
    base_name="$(basename "${csv_file%.csv}")"
    output_file="$output_dir/${base_name}_plot.png"
    echo "Plotting $(basename "$csv_file") -> $output_file"
    python3 "$plot_script" --input "$csv_file" --output "$output_file"
done

echo "Saved ${#csv_files[@]} plot(s) to $output_dir"
