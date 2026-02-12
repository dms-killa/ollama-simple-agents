# Vector Context Injection Guide

This guide explains how to use vector database context injection to enhance your agent workflows with relevant external knowledge.

## Overview

The vector context injection feature allows your agents to retrieve and use relevant information from a vector database during workflow execution. This is useful for:

- Grounding agent responses in existing documentation
- Providing domain-specific knowledge
- Reusing content from past work
- Ensuring consistency with style guides or templates
- Accessing up-to-date information not in the LLM's training data

## Architecture

```bash
User Request
     ↓
Flow Engine
     ↓
Step Configuration → Vector Context Enabled?
     ↓                        ↓
     No                      Yes
     ↓                        ↓
Execute Agent  →  Query Vector DB → Inject Context → Execute Agent
     ↓
Next Step
```

## Components

### 1. ContextManager (`core/context_manager.py`)

Handles all vector database operations:

- Connecting to vector databases (Chroma, Pinecone, Qdrant)
- Adding documents
- Querying for relevant context
- Formatting context for prompts

### 2. FlowEngine Enhancement (`core/flow_engine.py`)

Extended to:

- Initialize ContextManager
- Check if context retrieval is needed for each step
- Inject retrieved context into agent prompts

### 3. ContextStrategy

Determines when and how to retrieve context based on step configuration.

## Setup

### 1. Install Dependencies

For ChromaDB (recommended for local use):

```bash
pip install chromadb sentence-transformers
```

For Pinecone (cloud-based):

```bash
pip install pinecone-client
```

For Qdrant:

```bash
pip install qdrant-client
```

### 2. Configure Environment

Add to your `.env` file:

```env
# Vector Database Configuration
VECTOR_DB_TYPE=chroma                    # Options: chroma, pinecone, qdrant
CHROMA_PERSIST_DIR=./data/chroma_db     # For ChromaDB
EMBEDDING_MODEL=all-MiniLM-L6-v2        # Embedding model

# For Pinecone (if using)
PINECONE_API_KEY=your-api-key-here

# For Qdrant (if using)
QDRANT_URL=localhost
QDRANT_PORT=6333
```

### 3. Update main.py

The FlowEngine initialization now accepts vector context parameters:

```python
from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient

client = OllamaClient()
engine = FlowEngine(
    client=client,
    enable_vector_context=True,  # Enable context injection
    vector_db_type="chroma"      # Specify DB type
)
```

## Managing Your Vector Database

Use the `manage_vector_db.py` utility to populate and manage your collections.

### Add Documents

**From a single file:**

```bash
python manage_vector_db.py add knowledge_base.txt --collection blog_knowledge
```

**From a directory:**

```bash
python manage_vector_db.py add ./docs --collection technical_docs --pattern "*.md"
```

**From JSON:**

```bash
# JSON format: [{"content": "text1"}, {"content": "text2"}]
# or simply: ["text1", "text2"]
python manage_vector_db.py add data.json --collection my_collection
```

### List Collections

```bash
python manage_vector_db.py list
```

### Query a Collection

```bash
python manage_vector_db.py query "machine learning basics" --collection blog_knowledge --top-k 3
```

### Delete a Collection

```bash
python manage_vector_db.py delete old_collection
```

## Using Vector Context in Workflows

### YAML Configuration

Add vector context configuration to any step in your workflow:

```yaml
steps:
  - id: "research"
    name: "Topic Research with Context"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "research"
    
    # Vector context configuration
    vector_context_enabled: true                      # Enable for this step
    vector_collection: "blog_knowledge"               # Which collection to query
    vector_context_query: "background on {user_request}"  # Query template
    vector_top_k: 5                                   # Number of chunks to retrieve
    max_context_length: 3000                          # Optional: limit context size
```

### Configuration Options

| Parameter | Required | Default | Description |
| ----------- | ---------- | --------- | ------------- |
| `vector_context_enabled` | No | false | Enable context retrieval for this step |
| `vector_collection` | No | "default" | Name of the collection to query |
| `vector_context_query` | No | Uses input | Template for the query. Use `{user_request}` or `{step_name}` placeholders |
| `vector_top_k` | No | 5 | Number of relevant chunks to retrieve |
| `max_context_length` | No | None | Maximum characters for context (truncates if exceeded) |

### Query Templates

You can use placeholders in `vector_context_query`:

- `{user_request}` - The original user request
- `{step_output_key}` - Output from previous steps
- Any key from the workflow state

**Examples:**

```yaml
# Use user request directly
vector_context_query: "{user_request}"

# Combine with static text
vector_context_query: "best practices for {user_request}"

# Use previous step output
vector_context_query: "examples similar to {research}"

# Multiple placeholders
vector_context_query: "{user_request} with focus on {outline}"
```

## Example Workflows

### 1. Blog Writing with Knowledge Base

```yaml
name: "blog_with_context"
description: "Create blog posts grounded in existing knowledge"

steps:
  - id: "research"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "research"
    vector_context_enabled: true
    vector_collection: "blog_archive"
    vector_context_query: "previous blog posts about {user_request}"
    vector_top_k: 3

  - id: "outline"
    agent: "architect"
    model: "REASONING_MODEL"
    input_source: "research"
    output_key: "outline"
    vector_context_enabled: true
    vector_collection: "blog_templates"
    vector_top_k: 2
```

### 2. Technical Documentation

```yaml
name: "tech_doc_generation"
description: "Generate documentation with company standards"

steps:
  - id: "analyze"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "analysis"
    vector_context_enabled: true
    vector_collection: "api_docs"
    vector_context_query: "API documentation for {user_request}"

  - id: "write"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "analysis"
    output_key: "documentation"
    vector_context_enabled: true
    vector_collection: "doc_templates"
    vector_context_query: "documentation style guide and templates"
```

## Preparing Your Documents

### Format Requirements

Documents should be:

1. **Self-contained**: Each chunk should make sense independently
2. **Appropriately sized**: 200-1000 words per chunk works well
3. **Clean**: Remove unnecessary formatting, headers, footers

### Document Format Examples

**Plain Text (recommended for simplicity):**

```bash
Document chunk 1 with relevant information.
Can span multiple paragraphs.

Document chunk 2 with different topic.
Separated by blank lines.
```

**JSON (with metadata):**

```json
[
  {
    "content": "Document text here",
    "metadata": {
      "source": "blog_post_1",
      "date": "2024-01-15",
      "topic": "machine learning"
    }
  }
]
```

**Markdown:**

```markdown
## Topic 1
Content about topic 1

## Topic 2
Content about topic 2
```

## How It Works

1. **Query Building**: When a step has `vector_context_enabled: true`, the FlowEngine builds a query using the `vector_context_query` template or falls back to the step's input.

2. **Vector Search**: The ContextManager queries the specified collection and retrieves the top K most relevant chunks using semantic similarity.

3. **Context Formatting**: Retrieved chunks are formatted with headers and metadata for clarity.

4. **Injection**: The formatted context is prepended to the agent's input prompt:

   ```bash
   # Retrieved Context
   
   ## Source 1
   [relevant chunk 1]
   
   ## Source 2
   [relevant chunk 2]
   
   ---
   
   # Current Task
   [original input]
   ```

5. **Agent Execution**: The agent processes the enriched prompt with both retrieved context and the current task.

## Best Practices

### 1. Collection Organization

- Use separate collections for different types of knowledge
- Examples:
  - `blog_knowledge` - Past blog posts and research
  - `code_examples` - Code snippets and patterns
  - `style_guides` - Writing and formatting standards
  - `api_docs` - Technical documentation

### 2. Query Design

- Be specific in your query templates
- Use context from previous steps when relevant
- Consider what information would actually help the agent

### 3. Context Size

- Start with 3-5 chunks (`top_k`)
- Use `max_context_length` to prevent overwhelming the model
- Balance between enough context and maintaining focus

### 4. Step Selection

- Not every step needs vector context
- Enable it where external knowledge adds value:
  - Research phases
  - Template/example selection
  - Style consistency checks

### 5. Testing

- Start with a small, well-curated collection
- Test queries to ensure relevant results
- Adjust `top_k` based on result quality

## Programmatic Usage

### Adding Documents with Metadata

```python
from core.context_manager import ContextManager

# Initialize
cm = ContextManager(db_type="chroma")

# Add documents with metadata
documents = [
    "First document content",
    "Second document content"
]

metadatas = [
    {"source": "doc1.txt", "topic": "AI"},
    {"source": "doc2.txt", "topic": "ML"}
]

cm.add_documents(
    collection_name="my_collection",
    documents=documents,
    metadatas=metadatas
)
```

### Querying with Filters

```python
# Query with metadata filters
results = cm.query_context(
    query="machine learning basics",
    collection_name="my_collection",
    top_k=5,
    filter_metadata={"topic": "ML"}  # Only return ML documents
)
```

### Custom Formatting

```python
# Get raw results and format custom
results = cm.query_context(
    query="test query",
    collection_name="my_collection",
    top_k=3
)

# Custom formatting
formatted = cm.format_context_for_prompt(
    results,
    include_metadata=True,
    max_context_length=2000
)
```

## Troubleshooting

### "ChromaDB not installed"

```bash
pip install chromadb sentence-transformers
```

### "Collection not found"

Create the collection first:

```bash
python manage_vector_db.py add documents/ --collection my_collection
```

### No relevant results

- Check that documents are actually in the collection
- Try broader queries
- Verify the collection name matches

### Context too long

Set `max_context_length` in your workflow:

```yaml
vector_context_enabled: true
max_context_length: 2000  # Limit to 2000 chars
```

### Vector context not being used

1. Check that `enable_vector_context=True` in FlowEngine initialization
2. Verify `vector_context_enabled: true` in step configuration
3. Look for "Retrieved X context chunks" in console output

## Performance Considerations

- **Embedding Time**: First query may be slow as models load
- **Collection Size**: ChromaDB handles millions of documents efficiently
- **Query Speed**: Typically <100ms for most collections
- **Memory**: Embedding models require 100-500MB RAM

## Advanced Topics

### Custom Embedding Models

```python
# In .env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Multiple Vector DBs

```python
# Use different DBs for different purposes
chroma_engine = FlowEngine(client, vector_db_type="chroma")
pinecone_engine = FlowEngine(client, vector_db_type="pinecone")
```

### Hybrid Search

Combine vector search with keyword filtering:

```python
results = cm.query_context(
    query="machine learning",
    collection_name="docs",
    filter_metadata={"year": "2024"}
)
```

## Example: Complete Setup Workflow

```bash
# 1. Install dependencies
pip install chromadb sentence-transformers

# 2. Prepare documents
mkdir -p knowledge_base
echo "AI is the simulation of human intelligence..." > knowledge_base/ai_basics.txt
echo "Machine learning is a subset of AI..." > knowledge_base/ml_basics.txt

# 3. Add to vector DB
python manage_vector_db.py add knowledge_base/ --collection tech_knowledge --pattern "*.txt"

# 4. Verify
python manage_vector_db.py list
python manage_vector_db.py query "artificial intelligence" --collection tech_knowledge

# 5. Run workflow
python main.py --flow blog_workflow_with_context --request "Write about AI trends"
```

## Next Steps

- Start with a small collection and simple queries
- Monitor which contexts are being retrieved (check console output)
- Iterate on your query templates based on results
- Expand collections as you identify valuable knowledge sources
- Consider creating specialized collections for different agent roles
