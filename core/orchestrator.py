"""
Orchestrator support for conditional flow execution.

This module provides lightweight orchestrator capabilities for workflows -
allowing one agent to make decisions about which step to execute next,
whether to retry failed steps, and when to terminate the flow.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Decision(Enum):
    """Possible decisions an orchestrator can make."""
    CONTINUE = "continue"  # proceed to next step
    REPEAT = "repeat"       # repeat current/previous step
    BRANCH = "branch"       # jump to specific step by id
    TERMINATE = "terminate" # end the flow early


@dataclass
class OrchestratorDecision:
    """Result of an orchestrator agent's analysis."""
    decision: Decision
    target_id: Optional[str] = None  # for BRANCH - which step id to go to
    reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OrchestratorPrompt:
    """
    System prompt for an orchestrator agent.

    The orchestrator reads the full workflow state and decides what should happen next.
    It returns structured decisions that the flow engine can act upon.
    """

    BASE_PROMPT = """You are a workflow orchestrator for a multi-agent system. Your job is to analyze the current state of a workflow and decide what should happen next.

## Workflow Context
- **Flow Name**: {flow_name}
- **Current Step**: {current_step_id}
- **User Request**: {user_request}

## Current State (outputs from completed steps)
{state_snapshot}

## Completed Steps Summary
{completed_summary}

## Decision Options
You can make one of these decisions:

1. **CONTINUE** - Proceed to the next step in order (no target_id needed)
   Use when: The current step succeeded and everything looks good.

2. **REPEAT** - Re-run the previous/failed step (target_id = step id)
   Use when: A step produced poor output, failed, or needs retry.

3. **BRANCH** - Jump to a specific step by its id (target_id = step id)
   Use when: You want to skip ahead, go back multiple steps, or follow an alternative path.

4. **TERMINATE** - End the flow immediately (no target_id needed)
   Use when: Sufficient output has been produced, or further steps are pointless.

## Your Task
Analyze the state and return your decision in this exact format:

```json
{{
    "decision": "CONTINUE",
    "target_id": null,
    "reason": "Brief explanation of why you made this decision"
}}
```

Key considerations:
- Does the current output meet reasonable quality standards?
- Are there obvious errors or missing information?
- Is there enough state accumulated to conclude the workflow?
- Would revisiting a previous step improve results?

Only include the JSON object. No additional text."""

    def build_prompt(self, flow_name: str, current_step_id: str, user_request: str,
                     state: Dict[str, Any], completed_steps: List[Dict[str, Any]]) -> str:
        """Build the full prompt for the orchestrator agent."""
        # Create a concise state snapshot (max 5 keys to avoid context bloat)
        state_lines = []
        for key in list(state.keys())[:5]:
            value = state[key]
            preview = value[:200] + "..." if len(value) > 200 else value
            state_lines.append(f"- **{key}**: {preview}")
        state_snapshot = "\n".join(state_lines) if state_lines else "(empty)"

        # Create summary of completed steps
        step_summaries = []
        for step in completed_steps:
            status = "✓" if step.get("status") == "completed" else "✗"
            output_key = step.get("output_key", step.get("id", "unknown"))
            output_preview = ""
            if output_key and output_key in state:
                out_val = state[output_key]
                output_preview = f" ({len(out_val)} chars)"
            step_summaries.append(f"{status} {step.get('name', output_key)}{output_preview}")
        completed_summary = "\n".join(step_summaries) if step_summaries else "(no steps yet)"

        return self.BASE_PROMPT.format(
            flow_name=flow_name,
            current_step_id=current_step_id,
            user_request=user_request[:500] + "..." if len(user_request) > 500 else user_request,
            state_snapshot=state_snapshot,
            completed_summary=completed_summary,
        )

    @staticmethod
    def parse_decision(response: str) -> OrchestratorDecision:
        """Parse the orchestrator's JSON response into a decision."""
        import json

        # Extract JSON from markdown code blocks if present
        if "```json" in response:
            response = response.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in response:
            response = response.split("```", 1)[1].split("```", 1)[0]

        try:
            data = json.loads(response.strip())
            decision = Decision(data.get("decision", "continue"))
            return OrchestratorDecision(
                decision=decision,
                target_id=data.get("target_id"),
                reason=data.get("reason", ""),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse orchestrator decision: {e}")
            return OrchestratorDecision(
                decision=Decision.CONTINUE,
                reason=f"Parse error, defaulting to continue: {e}",
            )
