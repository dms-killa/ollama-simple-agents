#!/usr/bin/env python3
"""
create_collection.py

CLI tool to create and populate vector database collections.
Usage: python dev_tools/create_collection.py --name my_knowledge --docs "doc text 1 doc text 2"
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.context_manager import ContextManager
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(
        description='Create and populate vector database collections for agent workflows'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='my_knowledge',
        help='Collection name (default: my_knowledge)'
    )
    parser.add_argument(
        '--docs', '-d',
        type=str,
        nargs='+',
        default=[],
        help='Documents to add (space-separated or use --doc flag multiple times)'
    )
    parser.add_argument(
        '--doc',
        action='append',
        default=[],
        dest='docs_append',
        help='Add a single document (can be used multiple times)'
    )
    parser.add_argument(
        '--files',
        '-f',
        type=str,
        nargs='+',
        default=[],
        help='File paths to read and add as documents'
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=500,
        help='Chunk size in characters (default: 500)'
    )

    args = parser.parse_args()

    # Combine doc sources
    docs = args.docs + args.docs_append

    print("=" * 60)
    print(f"Vector Collection Creator")
    print(f"=" * 60)
    print(f"\nCollection: {args.name}")

    # Load env if exists
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv()

    # Initialize context manager
    db_type = __import__('os').getenv('VECTOR_DB_TYPE', 'chroma')
    print(f"Database: {db_type}")

    try:
        manager = ContextManager(db_type=db_type)
    except ImportError as e:
        print(f"\n✗ Error: Vector DB dependencies not installed")
        print(f"Install with: uv add chromadb sentence-transformers")
        if db_type == 'pinecone':
            print("For Pinecone: uv add pinecone-client")
        elif db_type == 'qdrant':
            print("For Qdrant: uv add qdrant-client")
        return 1

    # Get collection
    print(f"\nGetting/creating collection: {args.name}")
    try:
        collection = manager.get_or_create_collection(args.name)
        print(f"✓ Collection ready")
    except Exception as e:
        print(f"✗ Error accessing collection: {e}")
        return 1

    # Collect documents
    all_docs = []

    if docs:
        print(f"\nDirect documents ({len(docs)}):")
        for doc in docs:
            # Split by newlines or long spaces
            chunks = doc.split('\n')[:10]  # Limit to first 10 chunks
            all_docs.extend(chunks)

    if args.files:
        print(f"\nReading from files ({len(args.files)}):")
        for filepath in args.files:
            path = Path(filepath)
            if path.exists():
                text = path.read_text()
                chunks = text.split('\n')[:20]
                all_docs.extend(chunks)
                print(f"  ✓ {filepath}: {len(chunks)} chunks")
            else:
                print(f"  ✗ File not found: {filepath}")

    if not all_docs:
        print("\nNo documents to add.")
        print("Usage examples:")
        print("  python dev_tools/create_collection.py -n my_knowledge -d 'doc1 doc2 doc3'")
        print("  python dev_tools/create_collection.py -n product_docs -d 'Product A: features' --doc 'Product B: specs'")
        print("  python dev_tools/create_collection.py -n docs -f docs.md README.md")
        return 0

    # Add documents in chunks
    chunk_size = args.size
    added = 0

    for doc in all_docs:
        if len(doc.strip()) < chunk_size:
            collection.add([
                {'id': f'{args.name}_{added}', 'text': doc}
            ])
            added += 1

    # Count total
count = collection.count()
    print(f"\n" + "=" * 60)
    print("Collection Status")
    print("=" * 60)
    print(f"Collection: {args.name}")
    print(f"Documents added: {added} chunks")
    print(f"Total documents: {count}")

    if count == 0:
        print("\n⚠ Warning: Collection is empty!")
        print("The agent won't have any context to retrieve.")
    else:
        print("\n✓ Ready for RAG queries!")

    print(f"\nTo query this collection:")
    print(f"  python dev_tools/create_collection.py -n {args.name} --doc 'your query here'")


if __name__ == '__main__':
    main()
