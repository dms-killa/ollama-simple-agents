# Examples - Ready-to-Use Workflows

Collection of ready-to-use workflow configurations and sample requests.

## Table of Contents

1. [Chat Assistant](#chat-assistant)
2. [Code Review Assistant](#code-review-assistant)
3. [Customer Support QA](#customer-support-qa)
4. [Research Paper Summarizer](#research-paper-summarizer)
5. [Data Analysis Pipeline](#data-analysis-pipeline)
6. [Email Draft Assistant](#email-draft-assistant)
7. [Meeting Notes Generator](#meeting-notes-generator)
8. [Debugging Helper](#debugging-helper)
9. [Code Documentation Generator](#code-documentation-generator)
10. [Task Breakdown Assistant](#task-breakdown-assistant)

---

## Chat Assistant

A conversational agent that helps with various tasks.

### Workflow: Basic Chat

```yaml
name: "chat_assistant"
description: "Multi-turn conversation assistant"

steps:
  - id: "analyze_intent"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "intent_analysis"
    vector_context_enabled: true
    vector_collection: "chat_knowledge"
    vector_context_query: "conversation context about {user_request}"

  - id: "generate_response"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "intent_analysis"
    output_key: "response"
    context_keys:
      - "user_request"
```

### Sample Requests

```bash
# Simple greeting
python main.py --flow chat_assistant --request "Hello! How can I help?"

# Multi-turn conversation (use --state file for context)
python main.py --flow chat_assistant \
  --request "Explain machine learning in simple terms" \
  --output response.md
```

---

## Code Review Assistant

Reviews code for quality, bugs, and best practices.

### Workflow: Code Reviewer

```yaml
name: "code_reviewer"
description: "Automated code review with suggestions"

steps:
  - id: "analyze_code"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "code_analysis"

  - id: "identify_issues"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "code_analysis"
    output_key: "issues_found"
    context_keys:
      - "user_request"

  - id: "suggest_fixes"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "issues_found"
    output_key: "fix_suggestions"
```

### Sample Requests

```bash
# Review Python code
python main.py --flow code_reviewer \
  --request "Review this code for bugs and best practices:


def fibonacci(n):
    result = []
    a, b = 0, 1
    while len(result) < n:
        result.append(a)
        a, b = b, a+b
    return result

print(fibonacci(10))
"

# Security review
python main.py --flow code_reviewer \
  --request "Security audit of this authentication function:"
```

---

## Customer Support QA

Helps create and answer support questions based on knowledge base.

### Workflow: Support Assistant

```yaml
name: "support_assistant"
description: "Customer support Q&A with knowledge retrieval"

steps:
  - id: "retrieive_knowledge"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "support_context"
    vector_context_enabled: true
    vector_collection: "support_kb"
    vector_context_query: "answer to customer question about {user_request}"
    vector_top_k: 5

  - id: "draft_answer"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "support_context"
    output_key: "answer_draft"
    context_keys:
      - "user_request"

  - id: "review_tone"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "answer_draft"
    output_key: "final_answer"
```

### Sample Requests

```bash
# Answer product question
python main.py --flow support_assistant \
  --request "How do I reset my password?"

# Troubleshooting query
python main.py --flow support_assistant \
  --request "App crashes when clicking submit button. Error shows in console."
```

---

## Research Paper Summarizer

Summarizes and analyzes research papers or articles.

### Workflow: Paper Analyzer

```yaml
name: "paper_summarizer"
description: "Research paper summarization and analysis"

steps:
  - id: "extract_key_points"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "key_points"

  - id: "create_summary"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "key_points"
    output_key: "summary"

  - id: "extract_methodology"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "summary"
    output_key: "methodology_notes"
```

### Sample Requests

```bash
# Summarize a paper abstract
python main.py --flow paper_summarizer \
  --request "Summarize this research:

Abstract: We present a novel approach to natural language understanding
using transformer architectures. Our method achieves state-of-the-art
results on GLUE and SuperGLUE benchmarks while reducing computational
requirements by 40% compared to previous approaches.
"

# Extract main contributions
python main.py --flow paper_summarizer \
  --request "What are the main contributions of this paper?"
```

---

## Data Analysis Pipeline

Analyzes data and generates Python analysis scripts.

### Workflow: Data Analyst

```yaml
name: "data_analyst"
description: "Data analysis pipeline with code generation"

steps:
  - id: "understand_dataset"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "dataset_understanding"

  - id: "generate_analysis_code"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "dataset_understanding"
    output_key: "analysis_script"

  - id: "validate_approach"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "analysis_script"
    output_key: "final_report"
```

### Sample Requests

```bash
# Generate data exploration script
python main.py --flow data_analyst \
  --request "Create a Python script to analyze this CSV:
- Load customer_orders.csv
- Group by product_category
- Calculate avg order value per category
- Show top 5 categories by revenue
"

# Statistical analysis request
python main.py --flow data_analyst \
  --request "How should I perform A/B test analysis on this experiment?"
```

---

## Email Draft Assistant

Helps draft professional emails with appropriate tone.

### Workflow: Email Drafter

```yaml
name: "email_drafter"
description: "Professional email drafting assistant"

steps:
  - id: "analyze_context"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "email_context"

  - id: "draft_email"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "email_context"
    output_key: "email_draft"

  - id: "review_for_tone"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "email_draft"
    output_key: "final_email"
```

### Sample Requests

```bash
# Draft a thank you email
python main.py --flow email_drafter \
  --request "Draft a thank you email to my mentor for their advice on my job search"

# Write a follow-up
python main.py --flow email_drafter \
  --request "Help me write a follow-up email after submitting a job application"
```

---

## Meeting Notes Generator

Transcribes meeting topics and generates structured notes.

### Workflow: Meeting Notes AI

```yaml
name: "meeting_notes"
description: "Generate meeting notes from conversation topics"

steps:
  - id: "extract_topics"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "meeting_topics"

  - id: "create_agenda"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "meeting_topics"
    output_key: "agenda_items"

  - id: "generate_notes_template"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "agenda_items"
    output_key: "notes_template"
```

### Sample Requests

```bash
# Generate notes structure
python main.py --flow meeting_notes \
  --request "Weekly team sync agenda:
1. Review last week's sprint goals
2. Blockers discussion
3. Demo of new feature X
4. Action items assignment
"
```

---

## Debugging Helper

Helps diagnose and fix code bugs.

### Workflow: Code Debugger

```yaml
name: "code_debugger"
description: "Automated debugging assistance"

steps:
  - id: "analyze_error"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "error_analysis"

  - id: "generate_fix"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "error_analysis"
    output_key: "debug_solution"

  - id: "verify_fix"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "debug_solution"
    output_key: "verified_fix"
```

### Sample Requests

```bash
# Debug Python error
python main.py --flow code_debugger \
  --request "Debug this error:

File 'main.py', line 15, in calculate
divide by zero exception
Traceback (most recent call last):
  File "main.py", line 15, in <module>
    return safe_divide(a, b)
  ...
ZeroDivisionError: division by zero"
```

---

## Code Documentation Generator

Generates docstrings and API documentation.

### Workflow: Docs Generator

```yaml
name: "docs_generator"
description: "Generate documentation from code"

steps:
  - id: "analyze_functions"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "function_list"

  - id: "write_docstrings"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "function_list"
    output_key: "docstrings"

  - id: "format_output"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "docstrings"
    output_key: "formatted_docs"
```

### Sample Requests

```bash
# Generate docstrings
python main.py --flow docs_generator \
  --request "Add comprehensive docstrings to this module:

"""Utility functions for text processing."""
def clean_text(text):
    "Remove extra whitespace."
    return ' '.join(text.split())
```

---

## Task Breakdown Assistant

Breaks down complex tasks into actionable steps.

### Workflow: Task Planner

```yaml
name: "task_planner"
description: "Break down complex tasks into steps"

steps:
  - id: "analyze_requirements"
    agent: "analyst"
    model: "REASONING_MODEL"
    input_source: "user_request"
    output_key: "requirements"

  - id: "create_plan"
    agent: "developer"
    model: "CODING_MODEL"
    input_source: "requirements"
    output_key: "task_steps"

  - id: "estimate_effort"
    agent: "reviewer"
    model: "REASONING_MODEL"
    input_source: "task_steps"
    output_key: "estimated_plan"
```

### Sample Requests

```bash
# Break down project task
python main.py --flow task_planner \
  --request "Break down these requirements:
- Build a web app with user authentication
- Dashboard with charts
- Export to PDF reports
- Mobile-responsive design
"
```

---

## Quick Test Commands

Run quick tests on available workflows:

```bash
# Test all built-in workflows
python main.py --flow dev_workflow --request "Create a simple calculator app"
python main.py --flow blog_workflow_with_context --request "Write about AI safety"

# With quiet mode (only final output)
python main.py --flow dev_workflow \
  --request "Build X" \
  --quiet

# Save to file
python main.py --flow dev_workflow \
  --request "Create login system" \
  --output /tmp/login_system.md
```

## Tips for Best Results

1. **Be specific**: Include relevant context in your request
2. **Iterate**: Use output from one step as input to another
3. **Use context_keys**: Reference previous step outputs when helpful
4. **Enable vector context**: For knowledge-intensive tasks, enable RAG
5. **Check model capability**: Match task complexity to model size