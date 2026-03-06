# Ollama Agent System - Build Commands

# Python project with uv
#
# Usage:
#   make install-models      # Pull recommended models
#   make test                # Run test workflows
#   make clean-chroma        # Reset vector DB
#   make new-agent           # Generate agent template
#

.PHONY: help
help:
	@echo "Ollama Agent System Commands"
	@echo "============================="
	@echo ""
	@echo "Usage: make [COMMAND]"
	@echo ""
	@echo "Commands:"
	@echo "  make help              - Show this help message"
	@echo "  make install-models    - Pull recommended Ollama models"
	@echo "  make test              - Run all available workflows with test requests"
	@echo "  make clean-chroma      - Reset ChromaDB vector store (remove and recreate)"
	@echo "  make list-config       - Show current .env configuration values"
	@echo "  make new-agent         - Generate template files for a new agent"
	@echo "  make install-deps      - Install Python dependencies"
	@echo "  make setup             - Run full setup (deps + models)"

.PHONY: install-models
install-models:
	@echo "Pulling recommended Ollama models..."
	o llama pull llama3.1:8b || true
	o llama pull qwen2.5-coder:7b || true
	o llama pull mistral:7b || true
	@echo "Models installed! Run 'ollama list' to verify."

.PHONY: test
test:
	@echo "Running all available workflows with test requests..."
	python main.py --list-flows
	@echo ""
	python main.py --flow dev_workflow --request "Simple hello workflow test" --quiet
	python main.py --flow blog_workflow_with_context --request "Test blog post" --quiet

.PHONY: clean-chroma
clean-chroma:
	@echo "Clearing ChromaDB database..."
	rm -rf ./data/chroma_db/*
	@echo "Vector store cleared. Collections will be recreated on next use."

.PHONY: list-config
list-config:
	@echo "Current Configuration:"
	@echo "======================="
	@echo "OLLAMA_HOST=$(shell grep ^OLLAMA_HOST .env | cut -d= | cut -f2- || echo 'NOT SET')"
	@echo "REASONING_MODEL=$(shell grep ^REASONING_MODEL .env | cut -d= | cut -f2- || echo 'NOT SET')"
	@echo "CODING_MODEL=$(shell grep ^CODING_MODEL .env | cut -d= | cut -f2- || echo 'NOT SET')"
	@echo "VECTOR_DB_TYPE=$(shell grep ^VECTOR_DB_TYPE .env | cut -d= | cut -f2- || echo 'chroma')"
	@echo "CHROMA_PERSIST_DIR=$(shell grep ^CHROMA_PERSIST_DIR .env | cut -d= | cut -f2- || echo './data/chroma_db')"

.PHONY: new-agent
new-agent:
	@echo "Creating agent template files..."
	mkdir -p agents/templates/prompts
	echo "PLACEHOLDER_PROMPT" > agents/templates/prompts/new_agent.txt
	@echo "Created templates in agents/templates/"
	@echo "Edit the generated prompt and update config/flows/*.yaml"

.PHONY: install-deps
install-deps:
	uv sync

.PHONY: setup
setup: install-deps install-models
	@echo ""
	@echo "Setup complete! You can now run workflows."
	@echo "Try: python main.py --flow dev_workflow --request \"Hello\""
