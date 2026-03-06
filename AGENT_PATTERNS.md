# Agent Patterns

Guidelines for designing effective agent prompts and workflows.

## Core Agent Roles

This system uses four specialized agent roles, each with a distinct purpose:

| Role | Purpose | When to Use |
|------|---------|----|
| **analyst** | Requirements gathering, analysis, planning | Early pipeline stage, understanding the problem |
| **architect** | Design, structure, implementation strategy | After requirements, before coding |
| **developer** | Code generation, implementation | When actual work needs to be done |
| **reviewer** | Quality assurance, critique, refinement | Final stage or before deployment |

---

## Prompt Design Principles

### 1. Define Clear Boundaries

```text
# GOOD: Specific scope
"You are a Python code reviewer. Analyze the provided function for:
- Memory leaks from unclosed resources
- Performance issues (O(n²) loops on large data)
- Security vulnerabilities (SQL injection, XSS)
- Type safety and edge cases

Provide specific fix recommendations with code examples."

# BAD: Too vague  
"Review this code."
```

### 2. Provide Output Format

```text
"Output your analysis in this format:
1. Issues Found (numbered list)
   - Severity: [Critical/Warning/Suggestion]
   - Location: Line numbers
   - Issue description
   - Fix recommendation
2. Summary paragraph
3. Corrected code block"
```

### 3. Chain-of-Thought Prompting

```text
"Think step by step:
1. First, understand what the code is trying to accomplish
2. Next, analyze each component for issues
3. Then, determine which issues are critical vs minor
4. Finally, formulate fix recommendations

This ensures thorough analysis before output."
```

---

## Role-Specific Patterns

### Analyst Pattern

**Purpose:** Break down complex problems, identify requirements

```text
# System prompt for analyst:
"You are a requirements analyst. Your job is to:
1. Parse the user's request thoroughly
2. Identify implicit requirements not explicitly stated
3. Break the problem into logical subtasks
4. Output structured analysis in JSON format:
   {
     "intent": "What does the user want?",
     "requirements": ["list", "of", "requirements"],
     "constraints": ["technical", "business"],
     "subtasks": ["step-by-step", "plan"]
   }"
```

**Example Workflow:**
```yaml
- id: "analyze_requirements"
  agent: "analyst"
  input_source: "user_request"
  output_key: "requirements"
```

---

### Architect Pattern

**Purpose:** Design solutions, plan implementation

```text
# System prompt for architect:
"You are a system architect. Given the requirements:
1. Propose high-level design approach
2. Identify required components and their interfaces
3. Plan data flow between components  
4. Select appropriate patterns (MVC, microservices, etc.)
5. Note potential pitfalls and how to avoid them

Output a structured design document with diagrams described in text."
```

---

### Developer Pattern

**Purpose:** Generate code, implement solutions

```text
# System prompt for developer:
"You are an experienced Python developer. Implement the given requirements:
1. Write clean, documented, type-hinted code
2. Include error handling where appropriate
3. Add docstrings explaining purpose and usage
4. Follow PEP 8 style guidelines
5. Include example usage in docstring

For each file created:
- Name: filename.py
- Contains: [description]
```

---

### Reviewer Pattern

**Purpose:** Critique and improve work

```text
# System prompt for reviewer:
"You are a senior code reviewer. Review the provided work:
1. Check for correctness and edge cases
2. Verify code follows best practices
3. Identify security issues
4. Suggest improvements with specific examples
5. Rate overall quality (Excellent/Good/Fair/Poor)
6. Provide actionable feedback

Output format:
- Quality Score: X/10
- Strengths:
- Weaknesses:
- Specific Improvements:
- Revised Code (if applicable):"
```

---

## Context Injection Patterns

### Pattern 1: User Request Only

```yaml
- id: "first_step"
  input_source: "user_request"
  # Only sees original user prompt
```

### Pattern 2: Previous Step Output

```yaml
- id: "step_one"
  output_key: "analysis"
- id: "step_two" 
  input_source: "analysis"  # Uses step_one's output
```

### Pattern 3: Multiple Context Keys

```yaml
- id: "final_step"
  context_keys:
    - "original_request"    # Original user prompt
    - "requirements"        # From analyst step
    - "design"             # From architect step
    - "implementation"     # From developer step
```

### Pattern 4: Vector Context Injection

```yaml
- id: "grounded_step"
  vector_context_enabled: true
  vector_collection: "product_docs"
  vector_context_query: "Find documentation about {user_request}"
```

---

## Common Workflow Patterns

### Pattern: Review-after-Generation

A standard quality pattern:

```yaml
steps:
  - id: "generate"
    agent: "developer"
    output_key: "draft"
    
  - id: "review"  
    agent: "reviewer"
    input_source: "draft"
    context_keys:
      - "user_request"  # Review against original intent
```

---

### Pattern: Multi-Pass Refinement

Iterative improvement loop:

```yaml
steps:
  - id: "initial_draft"
    agent: "developer"
    output_key: "v1"
    
  - id: "critique"
    agent: "reviewer"
    input_source: "v1"
    output_key: "critique"
    
  - id: "improve"
    agent: "developer"
    input_source: "v1"   # Keep original for reference
    context_keys:
      - "critique"
    output_key: "v2"
```

---

### Pattern: Parallel Specialization

Using multiple analysts or reviewers:

```yaml
steps:
  - id: "req_analysis"
    agent: "analyst"
    output_key: "requirements"
    
  # Could add another analyst for constraints analysis
```

---

## When to Use Each Agent

| Scenario | Recommended Agents |
|----------|------|
| Simple Q&A | `analyst` + `developer` |
| Code generation | `analyst` (plan) → `developer` (implement) → `reviewer` (check) |
| Debugging | `analyst` (understand error) → `developer` (fix) → `reviewer` (verify) |
| Documentation | `developer` (draft) + `reviewer` (polish) |
| Research | `analyst` (extract info) → `developer` (synthesize) → `reviewer` (validate) |

---

## Prompt Engineering Tips

### Do's:
- Use specific examples in few-shot format
- Define output schema explicitly  
- Set boundaries on scope
- Provide context about domain knowledge needed
- Include error handling expectations

### Don'ts:
- Vague instructions like "do this well"
- Assumed knowledge not stated
- Multiple contradictory goals
- Open-ended without constraints
- Requests for external information (use vector context instead)

---

## Testing Your Prompts

```bash
# Test a specific agent's output
python main.py --flow test_flow --request "test" \
  | grep "output_key"
```

Use `CONFIG.md` to verify your model choices match the task complexity.

---

For more examples, see [`EXAMPLES.md`](./EXAMPLES.md).
For configuration details, see [`CONFIG.md`](./CONFIG.md).
