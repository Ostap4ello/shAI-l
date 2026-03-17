# ShAI - Minimal Shell Assistant Powered by Local LLM

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Ostap4ello/shAI-l
    cd shAI-l
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### CLI Commands
The application provides a command-line interface (CLI) for interacting with various functionalities.

```bash
py -m src -h
```

This module uses consists of submodules which provide own CLI commands, which are described below:

---

#### Module: `src.llm`

##### `generate`
Generate text responses from an LLM.
```bash
python -m src.llm generate --model <llm_model_name> "<prompt text>" --stream
```
- `--model`: Name of the model (e.g., "gpt-3.5-turbo"). Default: `qwen3:1.7b`.
- `--stream`: (optional) Enable streaming output.

##### `embed`
Generate embeddings for a given string.
```bash
python -m src.llm embed --model <llm_model_name> "<string to embed>"
```
- `--model`: Name of the embedding model (default: `ibm/granite-embedding:125m`).

##### `find`
Invoke the RAG pipeline to process a query.
```bash
python -m src.llm find "<query>"
```

---

#### Module: `src.db_retrieve`

##### `build`
Create an index from documents.
```bash
python -m src.db_retrieve build --db-path <path_to_documents> --index-path-within-db <index_name>
```
- `--db-path`: Path to the directory containing documents.
- `--index-path-within-db`: Name of the index subdirectory (default: `.index`).

##### `search`
Search previously created indexes for relevant data.
```bash
python -m src.db_retrieve search --top-k <int> "<search query>"
```
- `--top-k`: Number of top results to return.

##### `check`
Check whether an index exists.
```bash
python -m src.db_retrieve check --db-path <path_to_documents> --index-path-within-db <index_name>
```

---

#### Module: `src.rag`

##### `find`
Generate a RAG-enabled response for a query.
```bash
python -m src.rag find "<query>"
```
- `query`: The query string to process.

---

#### Module: `src.ext_utils`

##### `run_ollama`
Start the Ollama Docker container.
```bash
python -m src.ext_utils run_ollama --context-length <length> --gpus <gpu_list> --name <container_name>
```
- `--context-length`: Set the context length (default: `32000`).
- `--gpus`: Specify GPUs (default: all).
- `--name`: Docker container name (default: `ollama-node-1`).

##### `stop_ollama`
Stop the Ollama Docker container.
```bash
python -m src.ext_utils stop_ollama --name <container_name>
```
- `--name`: Docker container name (default: `ollama-node-1`).

##### `is_ollama_running`
Check whether the Ollama Docker container is running.
```bash
python -m src.ext_utils is_ollama_running --name <container_name>
```
- `--name`: Docker container name (default: `ollama-node-1`).

##### `convert_man_pages`
Convert `.groff` files in a directory to plain text.
```bash
python -m src.ext_utils convert_man_pages --src-dir <path_to_groff> --out-dir <path_to_output>
```
- `--src-dir`: Source directory containing `.groff` files.
- `--out-dir`: Target directory for converted `.txt` files.
