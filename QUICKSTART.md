# Quick Start Guide

Get your Ollama agent workflow running in 5 minutes.

## Prerequisites

- **Ollama installed** ([install](https://ollama.com/download))
- **Python 3.9+** installed
- Basic CLI knowledge

## Step 1: Setup Environment

```bash
# Copy the environment template
cp .env.example .env

# Edit with your models and settings
nano .env  # or vim/nano of your choice
```

### Minimum `.env` Configuration

```bash
OLLAMA_HOST=http://localhost:11434
REASONING_MODEL=llama3.1:8b
CODING_MODEL=qwen2.5-coder:7b
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
```

## Step 2: Pull Models

```bash
cd dev_tools
bash install_models.sh
```

Or manually:
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

## Step 3: Test Connection

```bash
python dev_tools/test_connection.py
```

Expected output:
```
✓ Ollama connected
✓ Models available: [list]
✓ Reasoning model test: success
```

## Step 4: Run Your First Workflow

```bash
# Simple chat bot
python main.py --flow chat_bot --request "Hello, who are you?"

# Code review assistant
python main.py --flow code_reviewer --request "Review this Python function for memory leaks"

# Research summarizer  
python main.py --flow research_assistant --request "Summarize: Attention Is All You Need (2017)"
```

## Step 5: Setup Vector Database (Optional)

```bash
python dev_tools/create_collection.py -c "my_knowledge_base" -d "doc1.txt doc2.txt doc3.txt"
```

Or use setup script:
```bash
python setup_example_collections.py
```

## Available Workflows

| Flow Name | Description |
|-----------|-------------|
| `chat_bot` | Conversational assistant |
| `code_reviewer` | Code analysis and suggestions |
| `research_assistant` | Paper summarization |
| `email_drafter` | Professional email drafting |
| `support_assistant` | Customer support Q&A |
| `debug_assistant` | Bug debugging and fixes |
| `data_analyst` | Data analysis pipeline |

## Common Commands

```bash
# List all available flows
python main.py --list-flows

# List installed Ollama models
python main.py --list-models

# Clean vector database
make clean-chroma

# Install all recommended models
make install-models
```

## Quick Troubleshooting

### "Model not found" error
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

### Connection timeout
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Update `OLLAMA_HOST` if using remote instance

### Vector DB errors
```bash
# Reset ChromaDB
rm -rf data/chroma_db/*
make clean-chroma
```

## Next Steps

1. Read CONFIG.md for full environment options
2. Browse EXAMPLES.md for workflow examples  
3. Check dev_tools/ for utility scripts
4. Customize prompts in prompts/

## Support

- Configuration help: See CONFIG.md
- Workflow examples: See EXAMPLES.md
- API reference: See API.md (coming soon)
- Agent patterns: See AGENT_PATTERNS.md (coming soon)

---

**That's it!** Your Ollama multi-agent system is ready to use.
