#!/usr/bin/env python3
"""
Multi-Agent Workflow System with Vector Context Injection

Entry point for running agent workflows with optional vector database context.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from core.ollama_client import OllamaClient
from core.flow_engine import FlowEngine

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description='Execute multi-agent workflows with Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic workflow
  python main.py --flow dev_workflow --request "Build a REST API"
  
  # With vector context enabled
  python main.py --flow blog_workflow_with_context --request "Write about AI"
  
  # Disable vector context for this run
  python main.py --flow blog_workflow_with_context --request "Quick post" --no-vector-context
  
  # List available workflows
  python main.py --list-flows
  
  # List Ollama models
  python main.py --list-models
        """
    )
    
    parser.add_argument(
        '--flow',
        type=str,
        help='Name of the workflow to execute (without .yaml extension)'
    )
    parser.add_argument(
        '--request',
        type=str,
        help='The user request/prompt for the workflow'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Optional: Save final output to file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show final output, suppress intermediate steps'
    )
    parser.add_argument(
        '--list-flows',
        action='store_true',
        help='List all available workflow configurations'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available Ollama models'
    )
    
    # Vector context options
    parser.add_argument(
        '--no-vector-context',
        action='store_true',
        help='Disable vector context injection for this run'
    )
    parser.add_argument(
        '--vector-db-type',
        type=str,
        default='chroma',
        choices=['chroma', 'pinecone', 'qdrant'],
        help='Vector database type (default: chroma)'
    )
    
    args = parser.parse_args()
    
    # Initialize Ollama client
    try:
        client = OllamaClient()
    except Exception as e:
        print(f"Error initializing Ollama client: {e}")
        return 1
    
    # Handle list-models command
    if args.list_models:
        try:
            models = client.list_models()
            print("\nAvailable Ollama Models:")
            print("-" * 40)
            for model in models:
                print(f"  - {model['name']}")
            print()
            return 0
        except Exception as e:
            print(f"Error listing models: {e}")
            return 1
    
    # Initialize flow engine with vector context support
    enable_vector = not args.no_vector_context
    engine = FlowEngine(
        client=client,
        enable_vector_context=enable_vector,
        vector_db_type=args.vector_db_type
    )
    
    # Handle list-flows command
    if args.list_flows:
        flows = engine.get_available_flows()
        print("\nAvailable Workflows:")
        print("-" * 40)
        for flow in flows:
            print(f"  - {flow}")
        print()
        return 0
    
    # Validate required arguments
    if not args.flow or not args.request:
        parser.print_help()
        print("\nError: Both --flow and --request are required for workflow execution")
        return 1
    
    # Execute workflow
    try:
        verbose = not args.quiet
        
        state = engine.run_flow(
            flow_name=args.flow,
            user_request=args.request,
            verbose=verbose
        )
        
        # Determine final output key (usually from last step)
        final_key = None
        flow_config = engine.load_flow(args.flow)
        if flow_config.get('steps'):
            final_key = flow_config['steps'][-1].get('output_key')
        
        final_output = state.get(final_key, '') if final_key else ''
        
        # Always show final output
        if args.quiet and final_output:
            print("\n" + "=" * 60)
            print("FINAL OUTPUT")
            print("=" * 60 + "\n")
            print(final_output)
        
        # Save to file if requested
        if args.output and final_output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(final_output)
            
            print(f"\n✓ Output saved to: {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUse --list-flows to see available workflows")
        return 1
    except Exception as e:
        print(f"Error executing workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())