import os
import subprocess
import logging

logger = logging.getLogger(__name__)

def _call_bash_script(script_path, *args) -> tuple[int, str, str]:
    logger.debug(f"Calling bash script: {script_path} with args: {args}")
    try:
        result = subprocess.run(
            [script_path, *args],
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.debug(f"Script {script_path} executed successfully.")
        logger.debug(f"stdout: {result.stdout}")
        logger.debug(f"stderr: {result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError as e:
        logger.error(f"Script not found: {script_path}")
        raise RuntimeError(f"Script not found: {script_path}") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing script: {script_path}. Return code: {e.returncode}, stdout: {e.stdout}, stderr: {e.stderr}")
        raise RuntimeError(f"Error executing script: {script_path}. Return code: {e.returncode}, stdout: {e.stdout}, stderr: {e.stderr}") from e

def run_ollama(context_length=32000, gpus="all", name="ollama-node-1"):
    logger.debug(f"Running Ollama Docker container with name: {name}, context_length: {context_length}, gpus: {gpus}")
    script_path = os.path.join(os.path.dirname(__file__), 'ollama-docker.sh')
    args = ["--name", name, "start", "--context-length", str(context_length), "--gpus", gpus]
    return _call_bash_script(script_path, *args)

def start_ollama(context_length=32000, gpus="all", name="ollama-node-1"):
    logger.debug(f"Starting Ollama Docker container with name: {name}, context_length: {context_length}, gpus: {gpus}")
    script_path = os.path.join(os.path.dirname(__file__), 'ollama-docker.sh')
    args = ["--name", name, "begin" ]
    try:
        return _call_bash_script(script_path, *args)
    except RuntimeError as e:
        logger.warning(f"Failed to start Ollama container. Trying to run it instead.")
        return run_ollama(context_length=context_length, gpus=gpus, name=name)

def stop_ollama(name="ollama-node-1"):
    logger.debug(f"Stopping Ollama Docker container with name: {name}")
    script_path = os.path.join(os.path.dirname(__file__), 'ollama-docker.sh')
    args = ["--name", name, "stop"]
    return _call_bash_script(script_path, *args)

def is_ollama_running(name="ollama-node-1") -> bool:
    logger.debug(f"Checking if Ollama Docker container with name: {name} is running")
    script_path = os.path.join(os.path.dirname(__file__), 'ollama-docker.sh')
    args = ["--name", name, "status"]
    return_code, stdout, stderr = _call_bash_script(script_path, *args)
    if stdout == "- Container is up\n- Ollama is up\n":
        logger.debug("Ollama container is running.")
        return True
    else:
        logger.debug("Ollama container is not running.")
        return False

def convert_man_pages_to_text(src_dir, out_dir):
    logger.debug(f"Converting Groff files from {src_dir} to {out_dir}")
    script_path = os.path.join(os.path.dirname(__file__), 'compile-groff.sh')
    args = ["-i", src_dir, "-o", out_dir]
    return _call_bash_script(script_path, *args)
