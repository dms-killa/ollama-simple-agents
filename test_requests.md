# Test Requests

Standard requests and edge cases for testing Ollama workflows.

## Simple Test Cases (Quick Smoke Tests)

These verify basic workflow functionality:

| Flow | Request | Expected Output |
|------|----|------|
| `chat_bot` | "Hello, how are you?" | Friendly response |
| `email_drafter` | "Write a polite follow-up email to a client about project status" | Structured email draft |
| `code_reviewer` | "Review this function: def add(a,b): return a+b" | Simple code review |
| `debug_assistant` | "What causes ValueError in Python?" | Clear explanation |

## Workflow-Specific Test Cases

### Chat Assistant

```bash
# Greeting test
python main.py --flow chat_bot --request "Hello"

# Knowledge question
python main.py --flow chat_bot --request "Explain photosynthesis"

# Opinion request  
python main.py --flow chat_bot --request "What's the best way to learn Python?"
```

### Code Reviewer

```bash
# Simple function
code_reviewer --request "Review this: def factorial(n): return 1 if n==0 else n*factorial(n-1)"

# Memory concern
code_reviewer --request "Review for memory leaks: results = []; for i in range(1000000): results.append(str(i))"

# Security check
code_reviewer --request "Security review: password = input() -> save to file"
```

### Debug Assistant

```bash
# Syntax error
python main.py --flow debug_assistant --request "Error: NameError: name 'x' is not defined"

# Logic error  
python main.py --flow debug_assistant --request "My loop doesn't terminate. Why?"

# Performance issue
python main.py --flow debug_assistant --request "This query is slow: SELECT * FROM users WHERE active=1"
```

### Research Assistant

```bash
# Paper summary
python main.py --flow research_assistant --request "Summarize: Transformer (Vaswani et al. 2017)"

# Concept explanation
python main.py --flow research_assistant --request "Explain attention mechanism in NLP"

# Literature search
python main.py --flow research_assistant --request "List papers on graph neural networks"
```

### Email Drafter

```bash
# Professional email
python main.py --flow email_drafter --request "Write an apology to a client for delayed delivery"

# Meeting invitation
python main.py --flow email_drafter --request "Draft calendar invite for project kickoff meeting"

# Thank you note
python main.py --flow email_drafter --request "Thank you email after successful presentation"
```

### Support Assistant

```bash
# Troubleshooting
python main.py --flow support_assistant --request "My internet keeps disconnecting"

# Feature question
python main.py --flow support_assistant --request "How do I enable two-factor authentication?"
```

## Edge Cases to Test

### Long Inputs

```bash
# 1000+ character request
echo "This is a very long user request that tests how the system handles large inputs without tokenization errors. [REPEAT FOR LENGTH]" | python main.py --flow chat_bot --request @- 
```

### Multi-Line Code

```bash
python main.py --flow code_reviewer --request \
"def complex_function(items, threshold=0.5):
    result = []
    for item in items:
        if item['score'] > threshold:
            result.append(item)
    return result"
```

### Special Characters

```bash
python main.py --flow email_drafter --request "Include these symbols: @ # & \* \` and unicode: \u2603\u2764\ud83d\ude0a"
```

### Ambiguous Requests

```bash
# Unclear intent  
python main.py --flow chat_bot --request "This one"

# Multiple interpretations
python main.py --flow email_drafter --request "Email about meeting"
```

### Empty/Missing Inputs

```bash
# Very short request
python main.py --flow chat_bot --request "x"

# Special query only
python main.py --flow chat_bot --request "???
```

## Performance Tests

### Latency Checks

```bash
# Measure time for simple flow
time python main.py --flow chat_bot --request "Hello" 2>&1 | grep real

# Time complex flow
time python main.py --flow dev_workflow --request "Build a full-stack todo app" 2>&1 | grep real
```

### Token Usage (estimate)

```bash
# Rough estimate using context length
# Simple chat: ~50-100 input tokens, ~200 output tokens
# Code generation: ~500-2000 input tokens for requirements, ~1000-3000 output for code
```

## Success Criteria by Flow

| Flow | Success Indicator |
|------|-------------------|
| `chat_bot` | Complete conversational response with no errors |
| `code_reviewer` | Issues found + fix suggestions in structured format |
| `debug_assistant` | Root cause explanation + corrected code block |
| `research_assistant` | Structured summary with methodology notes |
| `email_drafter` | Professional email with greeting, body, sign-off |
| `support_assistant` | Knowledge-grounded answer to customer question |

## Troubleshooting Test Failures

### "Model not found"
```
bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

### Connection errors
```
curl http://localhost:11434/api/tags
# Should return list of models if Ollama is running
```

### Timeout during test
```bash
# Increase timeout in .env
REQUEST_TIMEOUT=300
```

---

Use these test cases to validate your workflow setup and model configurations.
