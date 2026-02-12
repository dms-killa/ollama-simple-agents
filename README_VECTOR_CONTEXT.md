# Multi-Agent System with Vector Context Injection

A modular, extensible multi-agent workflow system using Ollama for local LLM execution, now enhanced with vector database context injection for knowledge-grounded agent responses.

## ✨ Key Features

- **Modular Agent System**: Define agents through simple text prompts
- **YAML Workflows**: Configure multi-step workflows without code changes
- **Vector Context Injection**: Ground agent responses in external knowledge bases
- **Flexible Architecture**: Support for multiple vector databases (ChromaDB, Pinecone, Qdrant)
- **Local LLM Execution**: Run everything locally with Ollama
- **Easy Document Management**: Simple CLI tools for managing knowledge bases

## 🆕 What's New: Vector Context Injection

Your agents can now access and utilize information from vector databases during workflow execution. This enables:

✅ **Knowledge-Grounded Responses** - Base outputs on existing documentation  
✅ **Consistency** - Ensure alignment with style guides and past work  
✅ **Domain Expertise** - Inject specialized knowledge per agent/step  
✅ **Reusability** - Leverage content from previous projects  
✅ **Up-to-Date Information** - Access current information beyond LLM training data

## 🏗️ Architecture

```bash
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Flow Engine    │─────▶│  Vector DB       │
└────────┬────────┘      │  (ChromaDB/etc)  │
         │               └──────────────────┘
         │                       │
         ▼                       │
    ┌────────┐                  │
    │ Step 1 │◀─────────────────┘
    └───┬────┘        Context Injection
        │
        ▼
    ┌────────┐
    │ Step 2 │
    └───┬────┘
        │
        ▼
    ┌────────┐
    │ Step N │
    └───┬────┘
        │
        ▼
  Final Output
```

## 📁 Project Structure

```bash
multi-agent-system/
├── main.py                          # CLI entry point
├── manage_vector_db.py              # Vector DB management utility
├── setup_example_collections.py     # Example setup script
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment configuration template
├── VECTOR_CONTEXT_GUIDE.md         # Comprehensive vector context guide
├── config/
│   └── flows/
│       ├── dev_workflow.yaml        # Original development workflow
│       ├── blog_workflow.yaml       # Original blog workflow
│       └── blog_workflow_with_context.yaml  # Example with vector context
├── agents/
│   └── base.py                      # Base agent class
├── prompts/
│   ├── analyst.txt                  # Analyst agent prompt
│   ├── architect.txt                # Architect agent prompt
│   ├── developer.txt                # Developer agent prompt
│   └── reviewer.txt                 # Reviewer agent prompt
├── core/
│   ├── ollama_client.py             # Ollama API client
│   ├── flow_engine.py               # Workflow execution (enhanced)
│   └── context_manager.py           # NEW: Vector DB integration
└── data/
    └── chroma_db/                   # ChromaDB storage (auto-created)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core system + ChromaDB (recommended for local use)
pip install -r requirements.txt

# OR for specific vector databases:
pip install chromadb sentence-transformers  # ChromaDB
pip install pinecone-client                 # Pinecone
pip install qdrant-client                   # Qdrant
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your settings
# At minimum, configure your Ollama models:
REASONING_MODEL=llama3.1:8b
CODING_MODEL=codellama:7b
```

### 3. Set Up Example Collections (Optional)

```bash
# Populate vector database with example documents
python setup_example_collections.py
```

This creates three example collections:

- `blog_knowledge` - AI/ML topic information
- `blog_templates` - Blog post structures
- `code_examples` - Python code patterns

### 4. Run Your First Workflow

```bash
# Without vector context (original functionality)
python main.py --flow dev_workflow --request "Build a task management API"

# With vector context (new!)
python main.py --flow blog_workflow_with_context \
    --request "Write a blog post about machine learning basics"
```

## 📚 Usage Examples

### Basic Workflow Execution

```bash
# List available workflows
python main.py --list-flows

# List available Ollama models
python main.py --list-models

# Run workflow with output saved to file
python main.py --flow dev_workflow \
    --request "Create a login system" \
    --output results.md

# Quiet mode (only final output)
python main.py --flow blog_workflow \
    --request "Write about Python" \
    --quiet
```

### Vector Database Management

```bash
# Add documents from a file
python manage_vector_db.py add knowledge_base.txt --collection tech_docs

# Add documents from a directory
python manage_vector_db.py add ./docs --collection my_docs --pattern "*.md"

# List all collections
python manage_vector_db.py list

# Query a collection
python manage_vector_db.py query "machine learning" --collection tech_docs --top-k 3

# Delete a collection
python manage_vector_db.py delete old_collection
```

### Advanced Usage

```bash
# Disable vector context for one run
python main.py --flow blog_workflow_with_context \
    --request "Quick note" \
    --no-vector-context

# Use different vector database
python main.py --flow my_workflow \
    --request "My task" \
    --vector-db-type pinecone
```

## 📝 Creating Workflows with Vector Context

### Basic Workflow (Without Context)

```yaml
name: "simple_workflow"
steps:
  - id: "analyze"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "analysis"
```

### Enhanced Workflow (With Context)

```yaml
name: "context_enhanced_workflow"
steps:
  - id: "research"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "research"
    
    # Vector context configuration
    vector_context_enabled: true
    vector_collection: "knowledge_base"
    vector_context_query: "background information about {user_request}"
    vector_top_k: 5
    max_context_length: 3000
```

### Configuration Options

| Parameter | Required | Default | Description |
| ----------- | ---------- | --------- | ------------- |
| `vector_context_enabled` | No | false | Enable context retrieval |
| `vector_collection` | No | "default" | Collection to query |
| `vector_context_query` | No | input text | Query template with placeholders |
| `vector_top_k` | No | 5 | Number of chunks to retrieve |
| `max_context_length` | No | None | Max characters for context |

## 🎯 Use Cases

### 1. Technical Documentation Generation

```yaml
# Query API docs to ensure accuracy
vector_collection: "api_documentation"
vector_context_query: "API endpoints for {user_request}"
```

### 2. Blog Writing with Past Content

```yaml
# Reference previous blog posts for consistency
vector_collection: "blog_archive"
vector_context_query: "previous articles about {user_request}"
```

### 3. Code Generation with Examples

```yaml
# Use code pattern library
vector_collection: "code_examples"
vector_context_query: "code patterns for {user_request}"
```

### 4. Style-Consistent Content

```yaml
# Apply style guides
vector_collection: "style_guides"
vector_context_query: "writing guidelines and templates"
```

## 🛠️ Preparing Your Documents

### Recommended Format

Documents should be:

- **Self-contained**: Each chunk makes sense independently
- **Appropriately sized**: 200-1000 words per chunk
- **Clean**: Remove unnecessary formatting

### Example Document Structure

**Plain text file** (`knowledge.txt`):

```bash
Topic: Machine Learning Basics

Machine learning is a subset of AI that enables systems
to learn from data...

Topic: Deep Learning

Deep learning uses neural networks with multiple layers...
```

**JSON file** (`knowledge.json`):

```json
[
  {
    "content": "Machine learning is...",
    "metadata": {
      "topic": "ML",
      "category": "basics"
    }
  }
]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
REASONING_MODEL=llama3.1:8b
CODING_MODEL=codellama:7b

# Vector Database
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Vector Database Options

**ChromaDB** (Local, recommended for development):

```env
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
```

**Pinecone** (Cloud-based):

```env
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-west1-gcp
```

**Qdrant** (Self-hosted or cloud):

```env
VECTOR_DB_TYPE=qdrant
QDRANT_URL=localhost
QDRANT_PORT=6333
```

## 📖 Documentation

- **[VECTOR_CONTEXT_GUIDE.md](VECTOR_CONTEXT_GUIDE.md)** - Complete guide to vector context injection
- **[CLAUDE.md](CLAUDE.md)** - Development guide for Claude Code
- **Workflow YAML files** - See `config/flows/` for examples

## 🤝 How It Works

1. **User submits request** via CLI
2. **Flow Engine loads workflow** configuration from YAML
3. For each step with `vector_context_enabled: true`:
   - **Query is built** using template or step input
   - **Vector DB is searched** for relevant chunks
   - **Context is formatted** with metadata
   - **Context is injected** into agent prompt
4. **Agent processes** enriched prompt (original input + context)
5. **Output is stored** in workflow state
6. **Next step** receives output + optional context

## 🎓 Best Practices

### Collection Organization

- Use separate collections for different knowledge types
- Examples: `docs`, `code_examples`, `style_guides`, `past_projects`

### Query Design

- Be specific in query templates
- Use context from previous steps: `{research}`, `{outline}`
- Match queries to the knowledge you need

### Context Management

- Start with 3-5 chunks (`top_k`)
- Use `max_context_length` to prevent overwhelming the model
- Enable context only where it adds value

### Testing

- Start with small, curated collections
- Test queries to ensure relevant results
- Monitor console output to see retrieved context

## 🐛 Troubleshooting

### Vector context not working

1. Check `enable_vector_context=True` in FlowEngine initialization
2. Verify `vector_context_enabled: true` in step config
3. Look for "Retrieved X context chunks" message

### No relevant results

- Verify collection exists: `python manage_vector_db.py list`
- Try broader queries
- Check collection has documents

### Performance issues

- Reduce `top_k` value
- Set `max_context_length` to limit size
- Consider using lighter embedding model

## 📦 Dependencies

**Core:**

- requests
- python-dotenv
- pyyaml

**Vector Database (choose one or more):**

- chromadb + sentence-transformers (recommended)
- pinecone-client
- qdrant-client

## 🌟 Examples

See complete examples in:

- `config/flows/blog_workflow_with_context.yaml` - Blog writing with knowledge base
- `setup_example_collections.py` - Sample data creation
- `VECTOR_CONTEXT_GUIDE.md` - Comprehensive usage guide

## 🚦 Roadmap

- [x] Vector context injection
- [x] Multiple vector DB support
- [x] Document management CLI
- [ ] Hybrid search (vector + keyword)
- [ ] Context caching
- [ ] Multi-modal context (images, code)
- [ ] Context ranking/reranking
- [ ] Automatic chunking strategies

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## 💬 Support

- **Issues**: GitHub Issues
- **Documentation**: See `VECTOR_CONTEXT_GUIDE.md`
- **Examples**: Check `config/flows/` directory

---

Built with ❤️ using Ollama and vector databases
