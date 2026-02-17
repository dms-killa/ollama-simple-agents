"""
Context strategy helpers for deciding when to retrieve vector context and how.
"""

class ContextStrategy:
    """Utility class providing static methods for workflow steps.

    The flow engine delegates to this class when a step declares
    ``vector_context_enabled: true``.  The class contains only very small
    pieces of logic so that the engine logic stays clean.
    """

    @staticmethod
    def should_retrieve_context(step: dict) -> bool:
        """Return True if a step wants vector context.

        The YAML step may include a boolean ``vector_context_enabled`` field.
        When omitted we treat it as ``False``.
        """
        return bool(step.get("vector_context_enabled", False))

    @staticmethod
    def build_context_query(step: dict, user_request: str, state: dict) -> str:
        """Format the query template for the step.

        The template is taken from ``step['vector_context_query']`` and
        rendered with ``user_request`` and the current state.  ``state`` may
        contain partial results from previous steps that some steps want to
        reference.
        """
        template = step.get("vector_context_query", "")
        try:
            return template.format(user_request=user_request, **state)
        except Exception:
            # Fall back to the template itself if formatting fails
            return template

    @staticmethod
    def get_collection_name(step: dict) -> str:
        """Return the collection name configured for the step.

        If the step does not specify one, an empty string is returned.
        """
        return step.get("vector_collection", "")
