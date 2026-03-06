# Multi-Agent Workflow System with Ollama

A modular, extensible multi-agent workflow system using Ollama for local LLM execution. Define multi-step workflows in YAML, each step powered by a specialized agent with its own system prompt. Supports optional vector database context injection via ChromaDB (with Pinecone and Qdrant stubs).

## Architecture

```
ollama-simple-agents/
├── main.py                          # CLI entry point
├── setup_example_collections.py     # Populate example vector DB collections
├── quick_start.sh                   # Automated setup script
├── config/
│   └── flows/                       # YAML workflow definitions
│       ├── dev_workflow.yaml
│       ├── blog_workflow.yaml
│       ├── blog_workflow_with_context.yaml
│       ├── thought_piece_workflow.yaml
│       └── writing_team_workflow.yaml
├── agents/
│   └── base.py                      # Base agent class
├── prompts/                         # System prompts for each agent
│   ├── analyst.txt                  # Requirements analysis
│   ├── architect.txt                # System design
│   ├── developer.txt                # Code implementation
│   ├── reviewer.txt                 # Code review
│   ├── vision_agent.txt             # Writing vision/thesis
│   ├── outline_agent.txt            # Writing outline
│   ├── section_agent.txt            # Section drafting
│   ├── revise_agent.txt             # Editorial revision
│   ├── researcher.txt               # Deep research
│   ├── drafter.txt                  # First-draft writing
│   ├── editor.txt                   # Structural editing
│   ├── fact_checker.txt             # Fact verification
│   └── de_slopper.txt               # AI slop removal
├── core/                            # Core system components
│   ├── ollama_client.py             # Ollama API client
│   ├── flow_engine.py               # Workflow execution engine
│   ├── context_manager.py           # Vector DB integration (Chroma, Pinecone, Qdrant)
│   └── context_strategy.py          # Context retrieval strategy helpers
├── tests/
│   └── test_step_types.py           # Pytest suite
└── projects/                        # Project dirs for project-aware workflows (runtime)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or use the automated setup script:
   ```bash
   chmod +x quick_start.sh && ./quick_start.sh
   ```

2. **Configure Ollama:**
   Create a `.env` file with your Ollama host and model aliases:
   ```env
   OLLAMA_HOST=http://localhost:11434
   REASONING_MODEL=llama3.1:8b
   CODING_MODEL=codellama:7b
   REQUEST_TIMEOUT=120
   ```
   Model aliases are any environment variable ending in `_MODEL`. Workflows reference aliases like `REASONING_MODEL` and the client resolves them to actual Ollama model names at runtime.

3. **Run a workflow:**
   ```bash
   python main.py --flow dev_workflow --request "Build a Python API for task management"
   ```

## Usage

```bash
# List available workflows
python main.py --list-flows

# List available Ollama models
python main.py --list-models

# Run with output saved to file
python main.py --flow dev_workflow --request "Create a login system" --output results.md

# Quiet mode (only final output)
python main.py --flow dev_workflow --request "Build a calculator" --quiet

# Project-aware writing workflow
python main.py --flow thought_piece_workflow --project my-essay --phase vision \
  --request "Write about the future of local AI"

# Full writing team with de-slopping
python main.py --flow writing_team_workflow --request "Write a blog post about why LLMs hallucinate"

# Disable vector context for a run
python main.py --flow blog_workflow_with_context --request "Quick post" --no-vector-context

# Use a specific vector database backend
python main.py --vector-db-type chroma
```

## Available Workflows

| Workflow | Description | Agents Used |
|----------|-------------|-------------|
| `dev_workflow` | Software development pipeline (analysis → design → implementation → review) | analyst, architect, developer, reviewer |
| `blog_workflow` | Content creation (research → outline → writing → editing) | analyst, architect, developer, reviewer |
| `blog_workflow_with_context` | Blog creation with vector DB context injection | analyst, architect, developer, reviewer |
| `thought_piece_workflow` | Iterative long-form writing with project state tracking | `{phase}_agent` (vision, outline, section, revise) |
| `writing_team_workflow` | Full writing team pipeline with de-slopping pass | researcher, drafter, editor, fact_checker, de_slopper |

## Step Types

Workflow steps have a `type` field that determines how they execute. Steps without an explicit type default to `agent`.

| Type | Description |
|------|-------------|
| `agent` (default) | Loads a system prompt, calls Ollama, stores the response. Supports `{key}` template placeholders. |
| `file_read` | Reads a file into state. Falls back to an optional `default` value if the file doesn't exist. |
| `file_write` | Writes content from state to a file, creating parent directories as needed. |
| `status_update` | Creates or updates a `STATUS.md` in a project directory (no LLM). Preserves the Key Terminology section. |
| `vector_embed` | Upserts content from state into a vector collection. Skipped if vector context is disabled. |

## How It Works

1. **Flow Engine** loads a YAML workflow definition from `config/flows/`
2. Each **step** specifies a type, agent, model, and I/O mapping
3. CLI parameters (`--project`, `--phase`) are merged into state for `{key}` template resolution
4. **State** is passed between steps via a shared dictionary
5. Agent steps optionally retrieve **vector context** from ChromaDB before calling Ollama
6. **Ollama Client** resolves model aliases (e.g. `REASONING_MODEL` → `llama3.1:8b`) and makes API calls
7. **Agents** are defined entirely by system prompts in `.txt` files — no code changes needed to add one

## Vector Context Injection

Steps can declare vector retrieval parameters to inject relevant context from a vector database before the LLM call:

```yaml
- id: "research"
  agent: "analyst"
  model: "REASONING_MODEL"
  input_source: "user_request"
  output_key: "research"
  vector_context_enabled: true
  vector_collection: "blog_knowledge"
  vector_context_query: "background information about {user_request}"
  vector_top_k: 5
  max_context_length: 3000
```

See [VECTOR_CONTEXT_GUIDE.md](VECTOR_CONTEXT_GUIDE.md) for full details.

## Creating Custom Workflows

1. Create a new YAML file in `config/flows/`:
   ```yaml
   name: "my_custom_flow"
   description: "What this workflow does"
   steps:
     - id: "step1"
       agent: "analyst"
       model: "REASONING_MODEL"
       input_source: "user_request"
       output_key: "analysis"
   ```

2. Add agent prompts in `prompts/` if needed (one `.txt` file per agent)

3. Run: `python main.py --flow my_custom_flow --request "..."`

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `*_MODEL` (e.g. `REASONING_MODEL`) | — | Model alias mappings. Any env var ending in `_MODEL` is available as an alias in workflow YAML. |
| `REQUEST_TIMEOUT` | `120` | Timeout in seconds for Ollama API requests |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Local directory for ChromaDB persistence |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |
| `PINECONE_API_KEY` | — | API key for Pinecone (if using that backend) |
| `QDRANT_URL` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

## Extending the System

**Adding new agents:** Create a `.txt` prompt file in `prompts/` and reference it by name in a workflow step.

**Adding new workflows:** Create a YAML file in `config/flows/` with steps that reference existing or new agents.

**Adding new step types:** Add a `_execute_<type>` handler method in `core/flow_engine.py` and a dispatch case in `execute_step`.

## Testing

```bash
python -m pytest tests/ -v
```
