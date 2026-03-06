# Configuration Reference

Complete guide to all environment variables, model options, and vector database configurations.

## Environment Variables

### Ollama Connection

| Variable | Default | Required | Description |
|--|--|--|--|
| `OLLAMA_HOST` | `http://localhost:11434` | Yes | URL of your Ollama server |
| `REQUEST_TIMEOUT` | `120` | No | API request timeout in seconds |

### Model Mappings

The system resolves model aliases automatically. Use either:
- **Alias**: `MYMODEL=actual-model-name`
- **Direct**: Use the actual model name directly in workflow YAML (e.g., `model: "llama3.1:8b"`)

| Variable | Purpose | Example |
|--|--|--|
| `REASONING_MODEL` | Analysis, design, review steps | `llama3.1:8b`, `gpt-oss-20b`, `mistral-nemo:12b` |
| `CODING_MODEL` | Code generation steps | `qwen2.5-coder:7b`, `codestral-latest:latest` |
| `GENERAL_MODEL` | Fallback for general tasks | `llama3.1:8b`, `mistral:7b` |

### Vector Database Settings

| Variable | Default | Description |
|--|--|--|
| `VECTOR_DB_TYPE` | `chroma` | Backend: `chroma`, `pinecone`, or `qdrant` |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model for text embeddings |
| `PINECONE_API_KEY` | *(unset)* | Your Pinecone API key |
| `QDRANT_URL` | `localhost` | Qdrant server URL |
| `QDRANT_PORT` | `6333` | Qdrant server port |

### Optional Settings

| Variable | Default | Description |
|--|--|--|
| `LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENABLE_VECTOR_CONTEXT` | `true` | Enable/disable vector DB context injection |

---

## Model Recommendations

### All-Local (Free, No API Costs)

```env
OLLAMA_HOST=http://localhost:11434
REASONING_MODEL=llama3.1:8b
CODING_MODEL=qwen2.5-coder:7b
GENERAL_MODEL=mistral:7b
```

**Models to pull:**
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull mistral:7b
```

### High Quality (Mixed Local + Cloud)

```env
REASONING_MODEL=claude-sonnet-3.5   # For analysis/review steps
CODING_MODEL=codestral-latest:latest # Code generation
GENERAL_MODEL=llama3.1:8b           # General tasks
```

### Cost-Optimized (Smaller Models)

```env
REASONING_MODEL=mistral:nemo:12b   # Good reasoning, faster
CODING_MODEL=codellama:7b          # Lightweight code model
GENERAL_MODEL=phi3:mini            # Very fast for simple tasks
```

---

## Vector Database Setup

### ChromaDB (Recommended for Local)

```env
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Usage:**
1. Collections are stored automatically in `CHROMA_PERSIST_DIR`
2. No setup required - just add documents using Python or the CLI tool
3. Ideal for local development and testing

### Pinecone (Production Cloud)

```env
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your-api-key-here
```

**Setup:**
1. Create a free account at https://www.pinecone.io
2. Get API key from project settings
3. Use Pinecone SDK to create collections and index documents

### Qdrant (Self-Hosted or Cloud)

```env
VECTOR_DB_TYPE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_PORT=6333
```

**Setup:**
1. Run Qdrant server (Docker recommended): `docker run -p 6333:6333 qdrant/qdrant`
2. Or use cloud hosting at https://qdrant.tech/cloud

---

## Example Configurations

### Development Environment

```env
# Use smaller models for fast iteration
OLLAMA_HOST=http://localhost:11434
REQUEST_TIMEOUT=60
REASONING_MODEL=llama3.1:8b
CODING_MODEL=qwen2.5-coder:7b
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
LOG_LEVEL=DEBUG  # See what's happening
ENABLE_VECTOR_CONTEXT=true
```

### Production Environment

```env
# Use high-quality models for production work
OLLAMA_HOST=http://ollama-server.company.local:11434
REQUEST_TIMEOUT=300
REASONING_MODEL=claude-sonnet-3.5
CODING_MODEL=codestral-latest:latest
GENERAL_MODEL=gpt-oss-20b
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=prod-key-here
LOG_LEVEL=WARNING
ENABLE_VECTOR_CONTEXT=true
```

### Minimal (No Vector Context)

```env
# Fastest option - skip RAG entirely
OLLAMA_HOST=http://localhost:11434
REASONING_MODEL=phi3:mini
CODING_MODEL=qwen2.5-coder:7b
ENABLE_VECTOR_CONTEXT=false
```

---

## Troubleshooting

### "Could not connect to Ollama"

```bash
# Check if Ollama is running
ollama list

# Start Ollama service (Linux)
systemctl status ollama

# Or start manually
ollama serve
```

### "Request timed out"

```env
# Increase timeout in .env
REQUEST_TIMEOUT=300
```

### "Collection not found" for vector DB

ChromaDB creates collections automatically. For Pinecone/Qdrant:

```bash
# Create a collection with your Python script
from pinecone import Pinecone
pc = Pinecone(api_key="...")
pc.create_index(name="my_knowledge", dimension=768, metric="cosine")
```

### No context retrieved from vector DB

1. Check that `vector_context_enabled: true` is set in your workflow YAML
2. Verify collection has documents: `python manage_vector.db list`
3. Test query relevance: `python manage_vector.db query "your query"`

---

## Creating Your First Collection

```bash
# Create a knowledge collection for your domain
python -c "
from context_manager import ContextManager
from dotenv import load_dotenv
import os

load_dotenv()

manager = ContextManager(db_type='chroma')
collection = manager.get_or_create_collection('product_docs')

# Add some documents (each is a text chunk)
docs = [
    'Product X features: feature1, feature2, feature3',
    'How to configure Product Y: step1, step2, step3',
    'Troubleshooting Product Z: error E1 means check network'
]

for doc in docs:
    collection.add([
        {'id': f'{hash(doc)}', 'text': doc}
    ])
"
```

See [manage_vector.db](manage_vector.db) for CLI collection management tools.