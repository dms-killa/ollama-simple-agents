# Multi-Agent System with Ollama

A modular, extensible multi-agent workflow system using Ollama for local LLM execution.

## Architecture

```
multi_agent_system/
├── main.py                 # CLI entry point
├── .env                    # Configuration (Ollama host, model mappings)
├── config/
│   └── flows/             # YAML workflow definitions
│       └── dev_workflow.yaml
├── agents/                # Agent implementations (modular)
│   ├── base.py           # Base agent class
├── prompts/              # System prompts for each agent
│   ├── analyst.txt
│   ├── architect.txt
│   ├── developer.txt
│   └── reviewer.txt
└── core/                 # Core system components
    ├── ollama_client.py  # Ollama API client
    └── flow_engine.py    # Workflow execution engine
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Ollama:**
   Edit `.env` to set your Ollama host and model mappings:
   ```env
   OLLAMA_HOST=http://localhost:11434
   REASONING_MODEL=llama3.1:8b
   CODING_MODEL=codellama:7b
   ```

3. **Run a workflow:**
   ```bash
   python main.py --flow dev_workflow --request "Build a Python API for task management"
   ```

## Usage

### List available flows:
```bash
python main.py --list-flows
```

### List available Ollama models:
```bash
python main.py --list-models
```

### Run with output saved to file:
```bash
python main.py --flow dev_workflow --request "Create a login system" --output results.md
```

### Quiet mode (only final output):
```bash
python main.py --flow dev_workflow --request "Build a calculator" --quiet
```

## Creating Custom Flows

1. Create a new YAML file in `config/flows/`:
   ```yaml
   name: "my_custom_flow"
   steps:
     - id: "step1"
       agent: "analyst"
       model: "REASONING_MODEL"
       input_source: "user_request"
       output_key: "analysis"
   ```

2. Add agent prompt in `prompts/` if needed

3. Run: `python main.py --flow my_custom_flow --request "..."`

## How It Works

The system uses a **pipeline architecture** where:

1. **Flow Engine** loads a YAML workflow definition
2. Each **step** specifies an agent, model, and I/O mapping
3. **State** is passed between steps via a shared dictionary
4. **Ollama Client** handles model resolution and API calls
5. **Agents** are defined by system prompts in `.txt` files

This design keeps the system modular—add new agents by creating prompt files, new workflows by creating YAML files, no changes to core code needed.