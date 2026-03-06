#!/bin/bash

# Quick Start Script for Vector Context Injection
# This script sets up everything needed to use vector context injection

echo "=================================="
echo "Vector Context Quick Start Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo "  ⚠ Please edit .env and configure your Ollama models"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies..."
echo "  This may take a few minutes..."
echo ""

pip install -q requests python-dotenv pyyaml chromadb sentence-transformers

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/chroma_db
mkdir -p config/flows
mkdir -p prompts
echo "✓ Directories created"
echo ""

# Set up example collections
echo "🗂️ Setting up example vector database collections..."
python3 setup_example_collections.py

if [ $? -eq 0 ]; then
    echo "✓ Example collections created"
else
    echo "⚠ Could not create example collections (this is optional)"
fi
echo ""

# Make scripts executable
chmod +x main.py manage_vector_db.py

echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your models in .env:"
echo "   nano .env"
echo ""
echo "2. Ensure Ollama is running:"
echo "   ollama serve"
echo ""
echo "3. Try an example workflow:"
echo "   python main.py --flow blog_workflow_with_context \\"
echo "       --request 'Write about machine learning basics'"
echo ""
echo "4. Add your own documents:"
echo "   python manage_vector_db.py add your_docs.txt --collection my_knowledge"
echo ""
echo "5. Read the complete guide:"
echo "   cat VECTOR_CONTEXT_GUIDE.md"
echo ""
echo "For help:"
echo "   python main.py --help"
echo "   python manage_vector_db.py --help"
echo ""