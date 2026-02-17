# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular, extensible multi-agent workflow system that uses Ollama for local LLM execution. The system allows users to define multi-step workflows using YAML configuration files, where each step is executed by a specialized agent with a specific role and prompt.

## Architecture

The system follows a pipeline architecture with these key components:

- **main.py**: CLI entry point that handles command-line arguments and orchestrates workflow execution
- **core/**: Core system components
  - `flow_engine.py`: Loads and executes workflow configurations defined in YAML files
  - `ollama_client.py`: Handles communication with Ollama API for LLM interactions
  - `context_manager.py`: Vector database integration (Chroma, Pinecone, Qdrant)
  - `context_strategy.py`: Helpers for deciding when/how to retrieve vector context
- **agents/**: Agent implementations (currently only base.py for agent interface)
- **prompts/**: System prompts for each agent role (analyst, architect, developer, reviewer, vision_agent, outline_agent, section_agent, revise_agent)
- **config/flows/**: YAML workflow definitions that specify the sequence of agent steps
- **projects/**: Project directories for project-aware workflows (created at runtime)
- **tests/**: Pytest suite covering step types and workflow plumbing

Additionally, the system supports optional **vector context injection** for steps that declare vector retrieval parameters. When a workflow or individual step specifies context keys such as `vector_top_k` or `vector_max_context_length`, the engine will query the configured vector store (Chroma, Pinecone, or Qdrant) and prepend relevant context to the agent prompt.

## Step Types

Workflow steps have a `type` field that determines how they execute. Steps without an explicit type default to `agent` for backward compatibility.

- **`agent`** (default): Loads a system prompt, calls Ollama, stores the response. Supports `{key}` template resolution in the `agent` field so workflows can be parameterised (e.g. `agent: "{phase}_agent"`).
- **`file_read`**: Reads a file into state. Falls back to an optional `default` value if the file doesn't exist.
- **`file_write`**: Writes content from state to a file, creating parent directories as needed.
- **`status_update`**: Creates or updates a `STATUS.md` in a project directory from flow state (no LLM involved). Preserves the Key Terminology section across runs.
- **`vector_embed`**: Upserts content from state into a vector collection for later retrieval. Gracefully skipped if vector context is disabled.

All step types support `{key}` template resolution in string fields such as `file_path`, `project_dir`, and `vector_collection`, where keys are resolved from the current flow state (including CLI params like `project` and `phase`).

## How It Works

1. Users define workflows in YAML files in `config/flows/`
2. Each workflow step specifies:
   - A `type` (`agent`, `file_read`, `file_write`, `status_update`, `vector_embed`) — defaults to `agent`
   - For agent steps: an agent role, model alias, input source, output key
   - For file steps: file paths with `{key}` template placeholders
   - Optional vector context parameters (`vector_top_k`, `vector_max_context_length`)
3. CLI parameters (`--project`, `--phase`) are merged into the initial state so YAML templates can reference them.
4. The FlowEngine loads the workflow and executes each step sequentially, resolving templates and injecting context where requested.
5. Each agent uses its specific system prompt from the `prompts/` directory.
6. State is passed between steps via a shared dictionary.

## Project-Aware Workflows

Writing projects live in `projects/<project_name>/`. Each project folder contains:
- **STATUS.md**: Tracks current phase, existing files, current focus, key terminology, and model config (YAML frontmatter). Written by the `status_update` step, read at the start of every workflow run.
- **`<phase>.md`** files: Output from each writing phase (vision, outline, section, revise).

The `thought_piece_workflow` uses `--project` and `--phase` CLI params to drive iterative long-form writing. Vector retrieval via Chroma is used during phases where grounding matters and is gracefully skipped if no collection exists yet.

## Available Workflows

- `dev_workflow.yaml`: Software development pipeline (analysis → design → implementation → review)
- `blog_workflow.yaml`: Content creation workflow (research → outline → writing → editing)
- `blog_workflow_with_context.yaml`: Blog workflow enhanced with vector context injection
- `thought_piece_workflow.yaml`: Iterative long-form writing (read status → write phase → save output → update status → embed for retrieval)

## Development Commands

- **Run a workflow**: `python main.py --flow dev_workflow --request "Build a Python API for task management"`
- **Run a thought piece**: `python main.py --flow thought_piece_workflow --project my-essay --phase vision --request "Write about X"`
- **List available flows**: `python main.py --list-flows`
- **List available Ollama models**: `python main.py --list-models`
- **Run with quiet mode**: `python main.py --flow dev_workflow --request "Build a calculator" --quiet`
- **Save output to file**: `python main.py --flow dev_workflow --request "Create a login system" --output results.md`
- **Disable vector context**: `python main.py --flow blog_workflow_with_context --request "Quick post" --no-vector-context`
- **Set vector database**: `python main.py --vector-db-type pinecone`
- **Run tests**: `python -m pytest tests/ -v`

## Configuration

The system uses environment variables defined in `.env`:
- `OLLAMA_HOST`: URL of the Ollama server (default: `http://localhost:11434`)
- `OLLAMA_MODEL_MAP`: Mapping of model aliases to Ollama model names (e.g., `REASONING_MODEL=llama3.1:8b`)
- `REQUEST_TIMEOUT`: Timeout for API requests (default: 120 seconds)
- `CHROMA_PERSIST_DIR`: Local directory for persisting Chroma embeddings (default: `./data/chroma_db`)
- `PINECONE_API_KEY`: API key for Pinecone if using that backend, etc.

## Extending the System

Adding new agents:
1. Create a new system prompt in `prompts/` (e.g., `researcher.txt`)
2. Add a new workflow step that references this agent

Adding new workflows:
1. Create a new YAML file in `config/flows/`
2. Define steps with type, agent, model, input/output mappings, and optional vector context parameters
3. Use `{key}` placeholders in string fields to parameterise steps with CLI params

Adding new step types:
1. Add a handler method `_execute_<type>` in `core/flow_engine.py`
2. Add the dispatch case in `execute_step`

## Key Files

- `main.py`: Entry point and CLI argument handling
- `core/flow_engine.py`: Workflow execution logic (agent, file_read, file_write, status_update, vector_embed steps)
- `core/ollama_client.py`: Ollama API interaction
- `core/context_manager.py`: Vector database integration
- `agents/base.py`: Base agent class
- `prompts/*.txt`: System prompts for different agent roles
- `config/flows/*.yaml`: Workflow definitions
- `tests/test_step_types.py`: Pytest suite for step types and workflow plumbing
