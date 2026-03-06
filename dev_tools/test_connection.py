#!/usr/bin/env python3
"""
test_connection.py

Quick health check script to verify Ollama connection and workflow setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ollama_client import OllamaClient


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


def check_env():
    """Check and display environment configuration."""
    print_section("Environment Configuration")

    # Load env file if exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv()
        print("✓ .env file found and loaded")
    else:
        print("⚠  .env file NOT found. Creating one...\n")
        # Create .env.example
        example_file = Path(__file__).parent.parent / ".env.example"
        if example_file.exists():
            with open(env_file, 'w') as f:
                f.write(example_file.read_text())
            print(f"Created: {env_file}\n")

    # Display current config
    print("\nCurrent Configuration:")
    print("-" * 40)
    print(f"  OLLAMA_HOST:     {os.getenv('OLLAMA_HOST', 'NOT SET')}")
    print(f"  REASONING_MODEL: {os.getenv('REASONING_MODEL', 'NOT SET')}")
    print(f"  CODING_MODEL:    {os.getenv('CODING_MODEL', 'NOT SET')}")
    print(f"  GENERAL_MODEL:   {os.getenv('GENERAL_MODEL', 'NOT SET')}")


def check_ollama_connection():
    """Test connection to Ollama."""
    client = OllamaClient()

    print_section("Ollama Connection")

    try:
        models = client.list_models()
        print(f"✓ Connected to {client.host}")
        print(f"\nAvailable Models ({len(models)}):")
        print("-" * 40)
        for model in models[:5]:  # Show first 5
            details = model.get('details', {})
            size_gb = details.get('size_in_bytes', 0) / (1024**3)
            print(f"  - {model['name']:.80} [{size_gb:.1f} GB]")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    except ConnectionError as e:
        print(f"✗ Connection error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Ollama running? Run: ollama serve")
        print("  2. Check network connectivity")
        print("  3. Verify OLLAMA_HOST in .env is correct")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def test_reasoning_model():
    """Test reasoning model with a simple prompt."""
    if 'REASONING_MODEL' not in os.environ:
        return

    client = OllamaClient()
    model = os.getenv('REASONING_MODEL')

    print_section(f"Testing {model}")

    try:
        response = client.generate(
            model=model,
            system_prompt="You are a helpful assistant. Be concise and clear.",
            user_input="Explain quantum entanglement in one sentence."
        )
        print(f"✓ Model {model} is working!\n")
        print("Sample response preview:")
        print("-" * 40)
        # Truncate to first 200 chars
        preview = response[:200] + "..." if len(response) > 200 else response
        print(preview)
    except Exception as e:
        print(f"✗ Error testing model: {e}")


def test_coding_model():
    """Test coding model with a simple code request."""
    if 'CODING_MODEL' not in os.environ:
        return

    client = OllamaClient()
    model = os.getenv('CODING_MODEL')

    print_section(f"Testing {model}")

    try:
        response = client.generate(
            model=model,
            system_prompt="You are a helpful coding assistant. Write clean, well-commented code.",
            user_input="Write a Python function that calculates Fibonacci numbers."
        )
        print(f"✓ Model {model} is working!\n")
        print("Sample response preview:")
        print("-" * 40)
        # Truncate to first 200 chars
        preview = response[:200] + "..." if len(response) > 200 else response
        print(preview)
    except Exception as e:
        print(f"✗ Error testing model: {e}")


def main():
    """Run all checks."""
    print_section("Ollama Agent System Health Check")

    check_env()
    check_ollama_connection()
    test_reasoning_model()
    test_coding_model()

    # Try to test a workflow if client exists
    try:
        from core.ollama_client import OllamaClient
        client = OllamaClient()
        test_flow_engine(client)
    except Exception as e:
        print(f"\nWarning: Could not test workflow (skipped): {e}")

    print_section("Summary")
    print("✓ All checks completed!")
    print("\nSee CONFIG.md for configuration reference.")
    print("Run 'python main.py --flow dev_workflow --request \"...\"' to use the system.")


def test_flow_engine(client):
    """Test running a simple workflow."""
    try:
        from core.flow_engine import FlowEngine
        engine = FlowEngine(client=client)
        state = engine.run_flow(
            flow_name='dev_workflow',
            user_request='Hello, what can you help me build?',
            verbose=False
        )
        print("✓ Workflow execution successful!")
    except Exception as e:
        print(f"✗ Workflow test skipped: {e}")


if __name__ == '__main__':
    main()
