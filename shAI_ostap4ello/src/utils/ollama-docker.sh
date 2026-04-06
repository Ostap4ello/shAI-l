#!/bin/bash
#
# A script to handle local ollama docker container management
#
# This script was adapted from Ostap4ello/dotfiles
# Copyright (c) [2026] Ostap4ello (okpeleh@gmail.com)
#

# defaults
context_length=32000
container_name="ollama-node-1"
container_gpus="all"

log() {
    echo "- $*"
}

get_base_url() {
    local url="http://127.0.0.1:11434/v1"

    if [ -n "$OPENAI_BASE_URL" ]; then
        url="$OPENAI_BASE_URL"
        [ -z "$OPENAI_API_KEY" ] && \
            log "Warning: OPENAI_API_KEY is not set. Requests may fail."
    fi

    echo "$url"
}

docker_action() {
    cmd="$1"
    shift
    pass_args=("$@")

    case "$cmd" in
        r|run)
            docker run -d \
                --name "$container_name" \
                -e OLLAMA_CONTEXT_LENGTH="$context_length" \
                -v ollama:/root/.ollama \
                -p 11434:11434 \
                --gpus="$container_gpus" \
                "${pass_args[@]}" \
                ollama/ollama
            ;;
        k|kill)
            docker rm -f "${pass_args[@]}" "$container_name"
            ;;
        a|attach)
            docker exec "${pass_args[@]}" -it "$container_name" /bin/bash
            ;;
        b|begin)
            docker start "${pass_args[@]}" "$container_name"
            ;;
        s|stop)
            docker stop "${pass_args[@]}" "$container_name"
            ;;
        c|cmd)
            docker exec -it "$container_name" /bin/ollama "${pass_args[@]}"
            ;;
        S|status)
            docker ps -f name="$container_name" --format '{{.Names}}' | grep -q "$container_name" && \
                log "Container is up." || \
                log "Container is down."

            curl -s "$(get_base_url)/models" > /dev/null && \
                log "Ollama is up." || \
                log "Ollama is down."
            ;;
    esac
    return $?
}


help_message() {
    cat << EOF
Usage: $0 [--name <name>] <command> [command-options]

Manage an Ollama Docker container.

Commands:
  r | run             Run container
    --context-length <length>  sets the context length (default: ${context_length})
    --gpus <gpus>              specify GPUs (default: ${container_gpus})
  k | kill            Kill and remove container
  a | attach          Attach to the container
  b | begin           Start container
  s | stop            Stop container

  c | command         Passes all arguments after that to "ollama" CLI command inside the 
                      container, e.g., "ollama-docker c ls" will do "ollama ls" inside the 
                      container.

Global Options:
  --name <name>               Container name (default: ${container_name})
  -h, --help                  Show this help message

Note:
  For all docker-related commands (r|k|a|b|s) you can pass additional docker flags
  after the command, e.g., docker flags.
EOF
}

main() {

    # Parse global flags before the command
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -n|--name)
                if [ -n "$2" ]; then
                    container_name="$2"
                    shift 2
                else
                    echo "Error: --name requires a non-empty argument."
                    exit 1
                fi
                ;;
            -h|--help)
                help_message
                exit 0
                ;;

            *)
                # Found the command
                break
                ;;
        esac
    done

    if [ "$#" -eq 0 ]; then
        echo "Error: No command provided. Use -h or --help for usage information."
        exit 1
    fi

    cmd="$1"
    shift

    case "$cmd" in
        r|run)
            # Parse run-specific flags
            pass_args=()
            while [ "$#" -gt 0 ]; do
                case "$1" in
                    --context-length)
                        if [ -n "$2" ]; then
                            context_length="$2"
                            shift 2
                        else
                            echo "Error: --context-length requires a non-empty argument."
                            exit 1
                        fi
                        ;;
                    --gpus)
                        if [ -n "$2" ]; then
                            container_gpus="$2"
                            shift 2
                        else
                            echo "Error: --gpus requires a non-empty argument."
                            exit 1
                        fi
                        ;;
                    *)
                        pass_args+=("$1")
                        shift
                        ;;
                esac
            done
            docker_action "$cmd" "${pass_args[@]}"
            ;;
        k|kill)
            ;&
        a|attach)
            ;&
        b|begin)
            ;&
        s|stop)
            ;&
        c|command)
            ;&
        S|status)
            docker_action "$cmd" "$@"
            ;;
        *)
            help_message
            exit 1
            ;;
    esac

    exit $?
}

# Main #
main "$@"
