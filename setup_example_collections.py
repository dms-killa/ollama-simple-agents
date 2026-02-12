#!/usr/bin/env python3
"""
Example setup for vector database context

This script creates example collections with sample documents
to demonstrate vector context injection.
"""

from pathlib import Path
from core.context_manager import ContextManager


def setup_blog_knowledge():
    """Create a sample blog knowledge collection."""
    
    documents = [
        """
        Artificial Intelligence (AI) Overview
        
        Artificial intelligence refers to the simulation of human intelligence in machines
        that are programmed to think and learn like humans. The field encompasses various
        approaches including machine learning, deep learning, and natural language processing.
        
        Key areas include computer vision, robotics, expert systems, and neural networks.
        AI systems are increasingly being used in healthcare, finance, transportation,
        and entertainment industries.
        """,
        
        """
        Machine Learning Fundamentals
        
        Machine learning is a subset of AI that enables systems to learn and improve from
        experience without being explicitly programmed. There are three main types:
        
        1. Supervised Learning: Training with labeled data
        2. Unsupervised Learning: Finding patterns in unlabeled data
        3. Reinforcement Learning: Learning through trial and error
        
        Common algorithms include linear regression, decision trees, random forests,
        neural networks, and support vector machines.
        """,
        
        """
        Deep Learning and Neural Networks
        
        Deep learning uses artificial neural networks with multiple layers to learn
        hierarchical representations of data. These networks are particularly effective
        for image recognition, natural language processing, and speech recognition.
        
        Popular architectures include:
        - Convolutional Neural Networks (CNNs) for image processing
        - Recurrent Neural Networks (RNNs) for sequential data
        - Transformers for language understanding
        
        Deep learning has revolutionized AI capabilities in recent years.
        """,
        
        """
        Natural Language Processing (NLP)
        
        NLP focuses on the interaction between computers and human language. It enables
        machines to understand, interpret, and generate human language in a valuable way.
        
        Applications include:
        - Text classification and sentiment analysis
        - Machine translation
        - Named entity recognition
        - Question answering systems
        - Chatbots and conversational AI
        
        Modern NLP relies heavily on transformer models like BERT and GPT.
        """,
        
        """
        AI Ethics and Responsible Development
        
        As AI systems become more powerful, ethical considerations become crucial.
        Key concerns include:
        
        - Bias and fairness in AI algorithms
        - Privacy and data protection
        - Transparency and explainability
        - Accountability for AI decisions
        - Job displacement and economic impact
        
        Organizations must develop AI responsibly with human values and societal
        impact in mind.
        """
    ]
    
    metadatas = [
        {"topic": "AI Overview", "category": "fundamentals"},
        {"topic": "Machine Learning", "category": "fundamentals"},
        {"topic": "Deep Learning", "category": "advanced"},
        {"topic": "NLP", "category": "applications"},
        {"topic": "AI Ethics", "category": "ethics"}
    ]
    
    return documents, metadatas


def setup_blog_templates():
    """Create sample blog templates collection."""
    
    documents = [
        """
        How-To Blog Post Template
        
        Title: How to [Achieve Goal]
        
        Introduction:
        - Hook with a common problem or question
        - Preview the solution you'll provide
        
        Main Content:
        - Step 1: [Clear action with explanation]
        - Step 2: [Next action with details]
        - Step 3: [Continue the sequence]
        
        Tips and Best Practices:
        - Pro tip #1
        - Common mistake to avoid
        
        Conclusion:
        - Recap key steps
        - Call to action
        """,
        
        """
        Technical Tutorial Template
        
        Title: Complete Guide to [Technology/Concept]
        
        Prerequisites:
        - Required knowledge
        - Tools needed
        
        Overview:
        - What you'll learn
        - Why it matters
        
        Core Concepts:
        - Concept 1: Definition and examples
        - Concept 2: Definition and examples
        
        Hands-On Example:
        - Code walkthrough
        - Explanation of key parts
        
        Advanced Topics:
        - Going deeper
        - Edge cases
        
        Resources:
        - Further reading
        - Related tutorials
        """,
        
        """
        Opinion/Analysis Blog Template
        
        Title: [Provocative Statement or Question]
        
        Hook:
        - Current event or trend
        - Why this matters now
        
        Thesis:
        - Your main argument in 1-2 sentences
        
        Supporting Points:
        - Evidence 1 with analysis
        - Evidence 2 with analysis
        - Evidence 3 with analysis
        
        Counter-Arguments:
        - Address opposing views
        - Explain why your position is stronger
        
        Implications:
        - What this means for the future
        - Call to action or reflection
        """
    ]
    
    metadatas = [
        {"type": "how-to", "format": "instructional"},
        {"type": "tutorial", "format": "technical"},
        {"type": "opinion", "format": "analytical"}
    ]
    
    return documents, metadatas


def setup_code_examples():
    """Create sample code examples collection."""
    
    documents = [
        """
        Python REST API Example with Flask
        
        ```python
        from flask import Flask, jsonify, request
        
        app = Flask(__name__)
        
        # Sample data
        tasks = [
            {'id': 1, 'title': 'Learn Python', 'done': False},
            {'id': 2, 'title': 'Build API', 'done': False}
        ]
        
        @app.route('/api/tasks', methods=['GET'])
        def get_tasks():
            return jsonify({'tasks': tasks})
        
        @app.route('/api/tasks', methods=['POST'])
        def create_task():
            task = {
                'id': len(tasks) + 1,
                'title': request.json['title'],
                'done': False
            }
            tasks.append(task)
            return jsonify({'task': task}), 201
        
        if __name__ == '__main__':
            app.run(debug=True)
        ```
        
        This shows basic CRUD operations with proper HTTP methods and status codes.
        """,
        
        """
        Error Handling Best Practice
        
        ```python
        import logging
        
        logger = logging.getLogger(__name__)
        
        def safe_divide(a, b):
            try:
                result = a / b
                return result
            except ZeroDivisionError:
                logger.error(f"Division by zero: {a} / {b}")
                return None
            except TypeError as e:
                logger.error(f"Type error in division: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        ```
        
        Always handle specific exceptions and log errors appropriately.
        """,
        
        """
        Async/Await Pattern in Python
        
        ```python
        import asyncio
        import aiohttp
        
        async def fetch_url(session, url):
            async with session.get(url) as response:
                return await response.text()
        
        async def fetch_multiple(urls):
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_url(session, url) for url in urls]
                return await asyncio.gather(*tasks)
        
        # Usage
        urls = ['http://example.com', 'http://example.org']
        results = asyncio.run(fetch_multiple(urls))
        ```
        
        Use async/await for I/O-bound operations to improve performance.
        """
    ]
    
    metadatas = [
        {"language": "python", "topic": "REST API", "framework": "Flask"},
        {"language": "python", "topic": "error handling"},
        {"language": "python", "topic": "async programming"}
    ]
    
    return documents, metadatas


def main():
    """Set up example collections."""
    
    print("Setting up example vector database collections...")
    print("=" * 60)
    
    # Initialize context manager
    cm = ContextManager(db_type="chroma")
    
    # Setup blog knowledge
    print("\n1. Creating 'blog_knowledge' collection...")
    docs, metas = setup_blog_knowledge()
    cm.add_documents(
        collection_name="blog_knowledge",
        documents=docs,
        metadatas=metas
    )
    print(f"   ✓ Added {len(docs)} documents")
    
    # Setup blog templates
    print("\n2. Creating 'blog_templates' collection...")
    docs, metas = setup_blog_templates()
    cm.add_documents(
        collection_name="blog_templates",
        documents=docs,
        metadatas=metas
    )
    print(f"   ✓ Added {len(docs)} templates")
    
    # Setup code examples
    print("\n3. Creating 'code_examples' collection...")
    docs, metas = setup_code_examples()
    cm.add_documents(
        collection_name="code_examples",
        documents=docs,
        metadatas=metas
    )
    print(f"   ✓ Added {len(docs)} code examples")
    
    # Verify setup
    print("\n" + "=" * 60)
    print("Setup complete! Available collections:")
    collections = cm.list_collections()
    for collection in collections:
        print(f"  - {collection}")
    
    # Test query
    print("\n" + "=" * 60)
    print("Testing query on 'blog_knowledge'...")
    results = cm.query_context(
        query="What is machine learning?",
        collection_name="blog_knowledge",
        top_k=2
    )
    
    print(f"\nFound {len(results)} relevant documents:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"  Topic: {result['metadata'].get('topic', 'N/A')}")
        print(f"  Preview: {result['content'][:100]}...")
    
    print("\n" + "=" * 60)
    print("✓ Vector database is ready to use!")
    print("\nNext steps:")
    print("  1. Try: python main.py --flow blog_workflow_with_context --request 'Write about AI'")
    print("  2. Add your own documents: python manage_vector_db.py add <file> --collection <name>")
    print("  3. Query collections: python manage_vector_db.py query '<query>' --collection <name>")
    print()


if __name__ == '__main__':
    main()