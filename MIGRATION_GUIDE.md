# Migration Guide: Adding Vector Context to Existing System

This guide helps you upgrade your existing multi-agent workflow system to include vector context injection capabilities.

## Overview of Changes

The vector context injection enhancement is **fully backward compatible**. Your existing workflows will continue to work exactly as before. The new features are opt-in on a per-step basis.

## What's New

### New Files Added

```bash
core/context_manager.py              # Vector DB integration
manage_vector_db.py                  # CLI for DB management
setup_example_collections.py         # Example data setup
VECTOR_CONTEXT_GUIDE.md             # Complete documentation
requirements.txt                     # Updated dependencies
.env.example                         # New config options
```

### Modified Files

```bash
core/flow_engine.py                  # Enhanced with context injection
main.py                              # Added vector context CLI options
```

### New Directories (Auto-Created)

```bash
data/chroma_db/                      # Vector database storage
```

## Migration Steps

### Step 1: Update Dependencies

Install the new vector database dependencies:

```bash
# For ChromaDB (recommended for local development)
pip install chromadb sentence-transformers

# OR for Pinecone (cloud-based)
pip install pinecone-client

# OR for Qdrant
pip install qdrant-client
```

### Step 2: Update Environment Configuration

Add new settings to your `.env` file:

```env
# Add these new lines to your existing .env

# Vector Database Configuration
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional: Pinecone settings (if using Pinecone)
# PINECONE_API_KEY=your-api-key
# PINECONE_ENVIRONMENT=us-west1-gcp

# Optional: Qdrant settings (if using Qdrant)
# QDRANT_URL=localhost
# QDRANT_PORT=6333
```

### Step 3: Update Your Code

#### Option A: Keep Existing Behavior (No Changes Required)

Your existing code will continue to work as-is:

```python
# This still works exactly as before
from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient

client = OllamaClient()
engine = FlowEngine(client=client)

# Run workflows as usual
state = engine.run_flow("dev_workflow", "Build an API")
```

#### Option B: Enable Vector Context Globally

```python
from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient

client = OllamaClient()

# NEW: Enable vector context for the engine
engine = FlowEngine(
    client=client,
    enable_vector_context=True,    # Enable context injection
    vector_db_type="chroma"         # Specify DB type
)

# Workflows now support vector context (if configured in YAML)
state = engine.run_flow("my_workflow", "My request")
```

#### Option C: Enable Per-Execution

```python
# Keep vector context disabled by default
engine = FlowEngine(client=client, enable_vector_context=False)

# But you can still use workflows that have vector context config
# (the context steps will just be skipped)
state = engine.run_flow("workflow_with_context", "Request")
```

### Step 4: Update Existing Workflows (Optional)

Your existing workflow YAML files require **no changes** to keep working. To add vector context to specific steps:

**Before (existing workflow):**

```yaml
steps:
  - id: "research"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "research"
```

**After (with vector context):**

```yaml
steps:
  - id: "research"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "research"
    # NEW: Add these lines to enable context for this step
    vector_context_enabled: true
    vector_collection: "knowledge_base"
    vector_context_query: "information about {user_request}"
    vector_top_k: 5
```

## Compatibility Matrix

| Component | Backward Compatible | Notes |
| ----------- | ------------------- | ------- |
| Existing workflows | ✅ Yes | Work without any changes |
| FlowEngine initialization | ✅ Yes | Old initialization still works |
| CLI commands | ✅ Yes | All existing commands unchanged |
| .env configuration | ✅ Yes | Old config still valid |
| Python API | ✅ Yes | Existing code requires no changes |

## Feature Comparison

### Before (Existing System)

```python
# Workflow step gets only the input
Input: "User request"
       ↓
    Agent processes
       ↓
    Output
```

### After (With Vector Context)

```python
# Workflow step can get input + relevant context
Input: "User request"
       ↓
Vector DB: [relevant doc 1, doc 2, doc 3]
       ↓
    Combined input = Context + User request
       ↓
    Agent processes enriched input
       ↓
    Output (better informed)
```

## Common Migration Scenarios

### Scenario 1: "I just want to keep using it as before"

**Action Required:** None!

Your system continues to work exactly as before. The new features are entirely optional.

### Scenario 2: "I want to try vector context on one workflow"

**Actions:**

1. Install ChromaDB: `pip install chromadb sentence-transformers`
2. Add documents: `python manage_vector_db.py add docs.txt --collection my_docs`
3. Create new workflow with `vector_context_enabled: true` for specific steps
4. Run with: `python main.py --flow my_new_workflow --request "..."`

### Scenario 3: "I want to upgrade all my workflows"

**Actions:**

1. Install dependencies
2. Add vector DB config to `.env`
3. Populate vector database with your knowledge base
4. Add `vector_context_enabled: true` to steps that would benefit
5. Test each workflow to verify improvements

### Scenario 4: "I want vector context but my documents aren't ready yet"

**Actions:**

1. Install dependencies
2. Update FlowEngine initialization with `enable_vector_context=True`
3. Set up workflows with `vector_context_enabled: true`
4. Steps will skip context retrieval if collections are empty
5. Add documents later: `python manage_vector_db.py add ...`

## Testing Your Migration

### 1. Verify Existing Workflows Still Work

```bash
# Your old workflows should run unchanged
python main.py --flow dev_workflow --request "Build something"
```

### 2. Test Vector Context Setup

```bash
# Set up example collections
python setup_example_collections.py

# Verify collections exist
python manage_vector_db.py list

# Test a query
python manage_vector_db.py query "test" --collection blog_knowledge
```

### 3. Try a New Workflow

```bash
# Run example workflow with context
python main.py --flow blog_workflow_with_context \
    --request "Write about machine learning"
```

### 4. Verify Console Output

When vector context is enabled and working, you should see:

```bash
✓ Vector context enabled using chroma
...
Executing: Topic Research with Context
  ℹ Retrieved 5 context chunks from 'blog_knowledge'
```

## Rollback Plan

If you need to rollback to the previous version:

### Quick Rollback (Disable Vector Context)

```python
# Just disable it in your code
engine = FlowEngine(client=client, enable_vector_context=False)
```

Or via CLI:

```bash
python main.py --flow my_workflow --request "..." --no-vector-context
```

### Full Rollback (Remove New Code)

1. Remove new files:

   ```bash
   rm core/context_manager.py
   rm manage_vector_db.py
   rm setup_example_collections.py
   ```

2. Restore original `flow_engine.py` and `main.py` from your backup

3. Uninstall vector DB packages:

   ```bash
   pip uninstall chromadb sentence-transformers
   ```

## Performance Impact

### Minimal Impact on Existing Workflows

If vector context is disabled (default for old workflows):

- **No performance change**
- **No memory change**
- **No storage change**

### Impact When Vector Context Enabled

- **First Query**: ~2-5 seconds (loading embedding model)
- **Subsequent Queries**: ~100-500ms per query
- **Memory**: +100-500MB (for embedding model)
- **Storage**: Depends on document collection size

## Troubleshooting Migration Issues

### "Module not found: chromadb"

**Solution:**

```bash
pip install chromadb sentence-transformers
```

### "Vector context not working"

**Check:**

1. Is it enabled in FlowEngine? `enable_vector_context=True`
2. Is it enabled in step? `vector_context_enabled: true`
3. Does the collection exist? `python manage_vector_db.py list`
4. Are there documents in the collection?

### "My old workflows are broken"

This shouldn't happen, but if it does:

1. Check for typos in YAML files
2. Verify `core/flow_engine.py` has the correct imports
3. Try with `--no-vector-context` flag
4. Review error messages for specifics

### "Performance is slower"

**Solutions:**

- Reduce `vector_top_k` value (default: 5)
- Add `max_context_length` limit
- Disable context for steps that don't need it
- Use lighter embedding model in `.env`

## Getting Help

1. **Read the docs:**
   - `VECTOR_CONTEXT_GUIDE.md` - Complete usage guide
   - `README_VECTOR_CONTEXT.md` - Overview and examples

2. **Try examples:**
   - `setup_example_collections.py` - Example data
   - `config/flows/blog_workflow_with_context.yaml` - Example workflow

3. **Test incrementally:**
   - Start with one collection
   - Add context to one step
   - Verify it works before expanding

## Next Steps After Migration

1. **Populate your knowledge base:**

   ```bash
   python manage_vector_db.py add your_docs/ --collection knowledge
   ```

2. **Experiment with queries:**

   ```bash
   python manage_vector_db.py query "your topic" --collection knowledge
   ```

3. **Update one workflow:**
   - Add `vector_context_enabled: true` to key steps
   - Test and measure improvements

4. **Iterate and expand:**
   - Add more collections
   - Refine query templates
   - Optimize `top_k` values

5. **Monitor and tune:**
   - Check console output for retrieved context
   - Verify context is relevant
   - Adjust configuration as needed

---

**Remember:** Migration is incremental and safe. You can adopt vector context features at your own pace, and your existing system continues to work throughout the process.
