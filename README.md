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
python -m shAI_ostap4ello.src.db build --db-path ./man-db
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
