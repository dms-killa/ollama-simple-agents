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
- **agents/**: Agent implementations (currently only base.py for agent interface)
- **prompts/**: System prompts for each agent role (analyst, architect, developer, reviewer)
- **config/flows/**: YAML workflow definitions that specify the sequence of agent steps

## How It Works

1. Users define workflows in YAML files in `config/flows/`
2. Each workflow step specifies:
   - An agent role (analyst, architect, developer, reviewer)
   - A model alias to use (resolved from environment variables)
   - Input source (from previous steps or user request)
   - Output key for storing results
3. The FlowEngine loads the workflow and executes each step sequentially
4. Each agent uses its specific system prompt from the `prompts/` directory
5. State is passed between steps via a shared dictionary

## Development Commands

- **Run a workflow**: `python main.py --flow dev_workflow --request "Build a Python API for task management"`
- **List available flows**: `python main.py --list-flows`
- **List available Ollama models**: `python main.py --list-models`
- **Run with quiet mode**: `python main.py --flow dev_workflow --request "Build a calculator" --quiet`
- **Save output to file**: `python main.py --flow dev_workflow --request "Create a login system" --output results.md`

## Configuration

The system uses environment variables defined in `.env`:
- `OLLAMA_HOST`: URL of the Ollama server (default: `http://localhost:11434`)
- Model mappings using `ALIAS_MODEL` format (e.g., `REASONING_MODEL=llama3.1:8b`)
- `REQUEST_TIMEOUT`: Timeout for API requests (default: 120 seconds)

## Extending the System

Adding new agents:
1. Create a new system prompt in `prompts/` (e.g., `researcher.txt`)
2. Add a new workflow step that references this agent

Adding new workflows:
1. Create a new YAML file in `config/flows/`
2. Define steps with agent, model, input/output mappings

## Key Files

- `main.py`: Entry point and CLI argument handling
- `core/flow_engine.py`: Workflow execution logic
- `core/ollama_client.py`: Ollama API interaction
- `agents/base.py`: Base agent class
- `prompts/*.txt`: System prompts for different agent roles
- `config/flows/*.yaml`: Workflow definitions