#!/usr/bin/env bash

#
# This script converts groff files in the chosen directory to ASCII text files
# using the groff command.
#
# Ostap4ello, 2025
#

set -eo pipefail

src_dir=
out_dir=

help() {
    echo "
    Usage: $0 [-h] -i <source_directory> -o <output_directory>
    
    This script converts groff files in the chosen directory to ASCII text files
    using the groff command.
    "
}

convert() {

    local backup_dir=".bak"

	if [[ ! -d "$src_dir" ]]; then
		echo "Directory '$src_dir' not found." >&2
		return 1
	fi

	shopt -s nullglob
	local files=("$src_dir"/*)
	shopt -u nullglob

	if [[ ${#files[@]} -eq 0 ]]; then
		echo "No files found in '$src_dir'." >&2
		return 1
	fi

	if [[ -d "$out_dir" ]]; then
        local backup_number=1
        local backup_base="${backup_dir}/${out_dir}.bak"
        while [[ -d "${backup_base}.${backup_number}" ]]; do
            ((backup_number++))
        done
        ((backup_number--))
        while [[ $backup_number -gt 0 ]]; do
            mv "${backup_base}.$backup_number" "${backup_base}.$((backup_number + 1))"
            ((backup_number--))
        done
        mv "$out_dir" "${backup_base}.1"
    fi

    mkdir -p "$out_dir"

	for file in "${files[@]}"; do
		if [[ -f "$file" ]]; then
            local base_name
            base_name="$(basename "$file")"
            local out_file="${out_dir}/${base_name}.txt"
            groff -Tascii -P -c -P -b -P -u -man "$file" > "$out_file"
		fi
	done
}

main() {
    while getopts "hi:o:" opt; do
        case "$opt" in
            h)
                help
                exit 0
                ;;
            i)
                src_dir="$OPTARG"
                ;;
            o)
                out_dir="$OPTARG"
                ;;
            *)
                help
                exit 1
                ;;
        esac
    done

    if [[ -z "$src_dir" || -z "$out_dir" ]]; then
        echo "Source and output directories must be specified." >&2
        help
        return 1
    fi

    convert
}

main "$@"