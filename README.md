# ShAI - Minimal Shell Assistant Powered by Local LLM

This is a minimalistic shell assistant that leverages local Large Language Models (LLMs) to provide various functionalities, such as generating text responses, creating embeddings, and performing Retrieval-Augmented Generation (RAG). The assistant is designed to be lightweight and easy to use, with a focus on local execution without relying on external APIs.

## Installation

### Install as a package (Recommended)
```bash
git clone https://github.com/Ostap4ello/shAI-l
cd shAI-l
pip install .
```

This will install the `shai` CLI command globally.

### System Dependencies
Install the following system dependencies:
- `docker` (for running Ollama). [Instructions](https://docs.docker.com/desktop/setup/install/linux/)  
    Run:
    ```bash
    docker pull ollama/ollama:latest
    ```
- `groff, jq` (for converting man pages)


## Configuration

ShAI uses a configuration file located at `~/.config/shai/config.conf`. The config file is created automatically with default values on first run.

### Configuration File Structure

```ini
[general]
keep_ollama_running = false

[llm]
model = qwen3:1.7b
embed_model = ibm/granite-embedding:125m
api_base_url = http://127.0.0.1:11434/v1
api_key = ollama

[db]
db_path = ~/.local/share/shai_db
index_path_within_db = .index
batch_size = 32
top_k = 5

[rag]
top_k = 5
model = qwen3:1.7b

[utils]
ollama_context_length = 32000
ollama_gpus = all
ollama_container_name = ollama-node-1
```

All settings can be overridden by:
1. Environment variables (for API settings: `OPENAI_API_KEY`, `OPENAI_BASE_URL`)
2. CLI arguments (e.g., `--model`, `--db-path`, `--top-k`)


## Usage

### CLI Commands
The application provides a command-line interface (CLI) for interacting with various functionalities.

If installed via pip:
```bash
shai -h
```

This module consists of submodules which provide their own CLI subcommands, which are described below:

---

### Available Commands

The main CLI `shai` has the following subcommands:
- `llm` - Direct LLM interactions (generation, embedding)
- `db` - Database indexing and retrieval
- `rag` - RAG-enabled generation
- `utils` - Miscellaneous utilities (Ollama management, man page conversion)


## Development

### Development Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/Ostap4ello/shAI-l
    cd shAI-l
    ```
2. Install python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Modules Directly
If you're developing without installing the package, you can run modules directly:

```bash
# Main CLI
python -m shAI_ostap4ello.src -h

# LLM module
python -m shAI_ostap4ello.src.llm generate "Hello world"
python -m shAI_ostap4ello.src.llm embed "Some text"

# Database module
python -m shAI_ostap4ello.src.db build --db-path ~/.local/share/shai_db
python -m shAI_ostap4ello.src.db search "search query"
python -m shAI_ostap4ello.src.db check

# RAG module
python -m shAI_ostap4ello.src.rag find "query"

# Utils module
python -m shAI_ostap4ello.src.utils start_ollama
python -m shAI_ostap4ello.src.utils stop_ollama
python -m shAI_ostap4ello.src.utils is_ollama_running
python -m shAI_ostap4ello.src.utils convert_man_pages --src-dir \<src\> --out-dir \<out\>
python -m shAI_ostap4ello.src.utils fetch_man_db
```

### Building the .tar.gz package
```bash
python -m build
```
This will create a `dist/` directory with the built package, which can be installed using pip:
```bash
pip install dist/\<package_name\>.tar.gz
```

## License

GPL-3.0-or-later
