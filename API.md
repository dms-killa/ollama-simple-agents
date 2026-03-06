# API Reference

Developer documentation for integrating Ollama workflows into other systems.

## Overview

This repository provides two main entry points:

1. **Standalone execution**: Use via CLI (`main.py`)
2. **Embedded integration**: Import `FlowEngine` directly into your application

---

## Core Classes

### OllamaClient

High-level client for interacting with Ollama API with automatic model alias resolution.

#### Usage Example

```python
from core.ollama_client import OllamaClient
import os

client = OllamaClient()

# Use model alias (resolved from env var)
response = client.generate(
    model=os.getenv('REASONING_MODEL'),
    prompt="Explain quantum entanglement simply",
    options={'temperature': 0.7, 'num_predict': 512}
)

print(response['response'])
```

#### Model Alias Resolution

```python
# .env file:
REASONING_MODEL=llama3.1:8b
CUSTOM_REASONING=mt-japanese-rusty:latest

# Access via alias:
client.generate(model='reasoning', prompt=...)
# Automatically resolves to: llama3.1:8b

client.generate(model='custom_reasoning', prompt=...)
# Automatically resolves to: mt-japanese-rusty:latest
```

#### Available Methods

| Method | Description |
|--------|-------------|
| `generate()` | Text generation with streaming support |
| `list_models()` | List all available models from Ollama |
| `get_model_info()` | Get detailed info about a specific model |
| `embed()` | Generate embeddings for vector DB |

---

### FlowEngine

Workflow execution engine that loads YAML configurations and orchestrates multi-agent pipelines.

#### Standalone Usage

```python
from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient

# Initialize engine
engine = FlowEngine(
    ollama_client=OllamaClient(),
    vector_db_type='chroma'  # optional
)

# Load and run workflow
result = engine.run(
    flow_name='code_reviewer',
    request="Review this function: def calc(n): return n * 2",
    output_key='fix_suggestions'
)

print(result)
```

#### Expected Output Structure

```python
{
    'steps': [
        {'id': 'parse_code', 'output': {...}},
        {'id': 'identify_issues', 'output': [...]},
        ...
    ],
    'final_output': 'fix_suggestions'  # or specified output_key
}
```

#### Embedded Integration Pattern

```python
# Integrate FlowEngine into your application class
class MyAgent:
    def __init__(self):
        self.engine = FlowEngine(OllamaClient())
    
    async def process_request(self, request: str) -> dict:
        # Run workflow asynchronously (using concurrent.futures)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self.engine.run,
                flow_name='code_reviewer',
                request=request
            )
            return future.result()
```

---

## Configuration Options

### FlowEngine Parameters

```python
FlowEngine(
    ollama_client: OllamaClient,                    # Required
    vector_db_type: str = 'chroma',                 # Chroma/Pinecone/Qdrant
    collection_name: str = None,                    # Default collection
    embedding_model: str = 'all-MiniLM-L6-v2',     # SentenceTransformer
    timeout: int = 120,                            # API request timeout
)
```

### OllamaClient Parameters

```python
OllamaClient(
    host: str = os.getenv('OLLAMA_HOST'),
    timeout: int = os.getenv('REQUEST_TIMEOUT', 120)
)
```

---

## Error Handling

```python
from core.ollama_client import OllamaConnectionError, ModelNotFoundError

try:
    response = client.generate(
        model='nonexistent_model',
        prompt="test"
    )
except ModelNotFoundError as e:
    print(f"Model not available: {e}")
    # Pull the model first
    ollama pull {model_name}
except OllamaConnectionError as e:
    print(f"Cannot reach Ollama: {e}")
```

---

## Async Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_workflow_async(flow_name: str, request: str) -> dict:
    engine = FlowEngine(OllamaClient())
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            executor,
            engine.run,
            flow_name, request
        )
        return await future
```

---

## State Management

Workflows maintain state across steps. Access intermediate results:

```python
result = engine.run(
    flow_name='dev_workflow',
    request="Build a todo app"
)

# Access any step's output
print(result['steps'][0]['output'])      # Analysis result
print(result['steps'][2]['output'])      # Implementation result
print(result['final_output'])            # Last step or specified key
```

---

## Workflow YAML Schema

```yaml
name: "workflow_name"
description: "What this workflow does"

steps:
  - id: "step_identifier"
    name: "Display Name (optional)"
    agent: "analyst|architect|developer|reviewer"  # Must have prompt file
    model: "MODEL_ALIAS"
    input_source: "user_request|previous_output_key"
    output_key: "result_name"
    
    # Optional: Context from previous steps
    context_keys:
      - "key1"
      - "key2"
    
    # Optional: Vector context injection
    vector_context_enabled: true
    vector_collection: "my_knowledge_base"
    vector_context_query: "Find info about {user_request}"
    vector_top_k: 5
```

---

## Testing Workflows

```python
from core.flow_engine import FlowEngine
from unittest.mock import MagicMock

# Test with mock Ollama responses
mock_response = {
    'response': 'Mock generated text',
    'done': True
}

engine = FlowEngine(
    ollama_client=MagicMock(return_value=mock_response),
    vector_db_type=None  # Disable for testing
)

result = engine.run('test_flow', 'Test request')
```

---

## Best Practices

1. **Cache responses**: For repeated requests, cache Ollama outputs
2. **Handle timeouts**: Set appropriate `REQUEST_TIMEOUT` for slow models
3. **Stream large responses**: Use streaming for long outputs
4. **Monitor token usage**: Track costs and limit per-request tokens
5. **Validate inputs**: Sanitize user requests before processing

---

## Example: Custom Workflow from Scratch

```python
from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient

# Define custom workflow programmatically
custom_workflow = {
    'name': 'custom_analysis',
    'description': 'My analysis pipeline',
    'steps': [
        {
            'id': 'summarize',
            'agent': 'analyst',
            'model': 'REASONING_MODEL',
            'input_source': 'user_request',
            'output_key': 'summary'
        },
        {
            'id': 'expand',
            'agent': 'developer', 
            'model': 'CODING_MODEL',
            'input_source': 'summary',
            'output_key': 'expanded'
        }
    ]
}

engine = FlowEngine(OllamaClient())
result = engine.run_workflow(custom_workflow, "What is machine learning?")
```

---

For more examples, see [`EXAMPLES.md`](./EXAMPLES.md) and [`CONFIG.md`](./CONFIG.md).
