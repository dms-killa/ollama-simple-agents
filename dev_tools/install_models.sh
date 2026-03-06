#!/bin/bash
#
# install_models.sh
# Helper script to pull recommended Ollama models for this agent system
#

set -e

echo "==========================="
echo "Ollama Model Installer"
echo "==========================="
echo ""

print_status() {
    echo "$1"
}

ask_confirm() {
    local prompt="$1"
    read -rp "${prompt} [y/N]:" answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

print_status "Recommended Models:"
echo ""
print_status "Option 1 - All-Local (Free)"
print_status "  REASONING_MODEL: llama3.1:8b          - Strong reasoning"
print_status "  CODING_MODEL: qwen2.5-coder:7b        - Code generation"
print_status "  GENERAL_MODEL: mistral:7b             - General tasks"
echo ""
print_status "Option 2 - High Quality (Cloud)"
print_status "  REASONING_MODEL: claude-sonnet-3.5    - Top-tier reasoning"
print_status "  CODING_MODEL: codestral-latest:latest  - Code-focused"
print_status "  GENERAL_MODEL: gpt-oss-20b            - General high-quality"
echo ""
print_status "Option 3 - Cost-Optimized (Smaller)"
print_status "  REASONING_MODEL: mistral:nemo:12b     - Good reasoning"
print_status "  CODING_MODEL: codellama:7b            - Lightweight code"
print_status "  GENERAL_MODEL: phi3:mini              - Very fast"
echo ""

# Pull models for selected configuration
read -rp "Which option to install? (1-4, or skip models): " choice

case "$choice" in
    1) 
        print_status "Pulling All-Local models..."
        ollama pull llama3.1:8b || true
        ollama pull qwen2.5-coder:7b || true
        ollama pull mistral:7b || true
        ;;
    2) 
        print_status "Pulling High Quality models..."
        ollama pull claude-sonnet-3.5 || true
        ollama pull codestral-latest:latest || true
        ollama pull gpt-oss-20b || true
        ;;
    3) 
        print_status "Pulling Cost-Optimized models..."
        ollama pull mistral:nemo:12b || true
        ollama pull codellama:7b || true
        ollama pull phi3:mini || true
        ;;
    *) 
        if [[ "$choice" != "skip" ]]; then
            echo ""
            read -rp "Enter models to pull (space-separated): " custom_list
            for model in $custom_list; do
                ollama pull "$model" || true
            done
        fi
        ;;
esac

echo ""
echo "==========================="
echo "Installation Complete"
echo "==========================="
echo ""
echo "Next steps:"
echo "  1. Edit .env to configure your model aliases"
echo "  2. Run: python main.py --list-models"
