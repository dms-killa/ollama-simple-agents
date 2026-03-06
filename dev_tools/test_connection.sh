#!/bin/bash
#
# test_connection.sh
# Quick health check for Ollama agent system
#

echo ""
echo "===================================="
echo "Ollama Agent System Health Check"
echo "===================================="
echo ""

# Navigate to repo root if run from tools directory
cd /home/parker/mnt/git_repos/ollama-simple-agents

# Run Python health check
python dev_tools/test_connection.py || true
