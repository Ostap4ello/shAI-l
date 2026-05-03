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

extract-and-convert() {

    if [[ ! -d "$src_dir" ]]; then
        echo "Directory '$src_dir' not found." >&2
        return 1
    fi

    echo "Listing $src_dir"
    declare -a folders
    declare -a files
    files=()
    folders=("$src_dir")
    while [[ ${#folders[@]} -gt 0 ]]; do
        local current_folder="${folders[0]}"
        folders=("${folders[@]:1}")

        for entry in "$current_folder"/*; do
            if [[ -d "$entry" ]]; then
                folders+=("$entry")
            elif [[ -f "$entry" ]] && [[ "$entry" == *.gz ]]; then
                files+=("$entry")
            fi
        done
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No files found in '$src_dir'." >&2
        return 1
    fi

    if [[ -d "$out_dir" ]] && [[ ! -n "$(ls -A "$out_dir")" ]]; then
        local backup_number=1
        local backup_base="${out_dir}.bak"

        echo "Creating backup for $out_dir in $backup_base.1"
        while [[ -d "${backup_base}.${backup_number}" ]]; do
            ((backup_number++))
        done
        ((backup_number--))
        if [[ $backup_number -gt 0 ]]; then
            echo "Found $backup_number of other backups. Incrementing each"
        fi
        while [[ $backup_number -gt 0 ]]; do
            mv "${backup_base}.$backup_number" "${backup_base}.$((backup_number + 1))"
            ((backup_number--))
        done
        mv "$out_dir" "${backup_base}.1"
    fi

    mkdir -p "$out_dir"

    tmp_dir=$(mktemp -d --suffix "man-")

    echo "Decompressing pages"
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            local out_file="${tmp_dir}/${file#$src_dir/}"
            if [[ -L "$file" ]]; then
                # copy the target of the symlink
                cp -L "$file" "$file.tmp"
                mv "$file.tmp" "$file"
            fi

            echo "  - decompressing $file to $out_file"
            mkdir -p "$(dirname "$out_file")"
            gzip --decompress --keep "$file" --stdout > "$out_file"
        fi
    done

    echo "Compiling pages"
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            local src_file="${tmp_dir}/${file#$src_dir/}"
            local out_file="${out_dir}/${file#$src_dir/}"
            out_file=${out_file%.gz}.txt

            echo "  - compiling $src_file to $out_file"
            mkdir -p "$(dirname "$out_file")"
            groff -Tascii -P -c -P -b -P -u -man "$src_file" > "${out_file}"
        fi
    done

    echo "Done. Cleaning"
    rm -rf "$tmp_dir"
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

    extract-and-convert
}

main "$@"
