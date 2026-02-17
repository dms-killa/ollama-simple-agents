import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from core.ollama_client import OllamaClient
from core.context_manager import ContextManager, ContextStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowEngine:
    """
    Executes multi-step agent workflows with vector database context injection.

    Enhanced to support retrieval of relevant context from vector databases
    before executing each step, with improved error handling.
    """

    def __init__(
        self, 
        client: OllamaClient, 
        flows_dir: str = "config/flows",
        prompts_dir: str = "prompts",
        enable_vector_context: bool = True,
        vector_db_type: str = "chroma",
        strict_mode: bool = False
    ):
        """
        Initialize the flow engine.

        Args:
            client: OllamaClient instance
            flows_dir: Directory containing workflow YAML files
            prompts_dir: Directory containing agent prompt files
            enable_vector_context: Whether to enable vector DB context retrieval
            vector_db_type: Type of vector database to use
            strict_mode: If True, raise exceptions on context retrieval failures
        """
        self.client = client
        self.flows_dir = Path(flows_dir)
        self.prompts_dir = Path(prompts_dir)
        self.state = {}
        self.strict_mode = strict_mode

        # Initialize context manager if enabled
        self.context_manager = None
        self.vector_context_enabled = enable_vector_context

        if enable_vector_context:
            try:
                self.context_manager = ContextManager(db_type=vector_db_type)
                logger.info(f"✓ Vector context enabled using {vector_db_type}")
            except ImportError as e:
                logger.warning(f"⚠ Vector context disabled: {e}")
                self.vector_context_enabled = False
            except (ConnectionError, ValueError) as e:
                if strict_mode:
                    raise
                logger.warning(f"⚠ Vector context initialization failed: {e}")
                self.vector_context_enabled = False
            except Exception as e:
                if strict_mode:
                    raise
                logger.error(f"Unexpected error initializing context manager: {e}")
                self.vector_context_enabled = False

    def load_flow(self, flow_name: str) -> Dict[str, Any]:
        """
        Load a workflow configuration from YAML.

        Args:
            flow_name: Name of the flow file (without .yaml extension)

        Returns:
            Dictionary containing the flow configuration

        Raises:
            FileNotFoundError: If the flow file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        flow_path = self.flows_dir / f"{flow_name}.yaml"

        if not flow_path.exists():
            raise FileNotFoundError(
                f"Flow '{flow_name}' not found at {flow_path}"
            )

        with open(flow_path, 'r') as f:
            return yaml.safe_load(f)

    def load_system_prompt(self, agent_name: str) -> str:
        """
        Load the system prompt for a specific agent.

        Args:
            agent_name: Name of the agent (corresponds to prompt file name)

        Returns:
            The system prompt text

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        prompt_path = self.prompts_dir / f"{agent_name}.txt"

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt for agent '{agent_name}' not found at {prompt_path}"
            )

        with open(prompt_path, 'r') as f:
            return f.read()

    def retrieve_vector_context(
        self,
        step: Dict[str, Any],
        user_request: str
    ) -> Optional[str]:
        """
        Retrieve relevant context from vector database for this step.

        Args:
            step: The current workflow step configuration
            user_request: The original user request

        Returns:
            Formatted context string, or None if not applicable/failed

        Raises:
            Exception: In strict_mode, propagates context retrieval errors
        """
        # Check if context retrieval is enabled globally and for this step
        if not self.vector_context_enabled:
            return None

        if not ContextStrategy.should_retrieve_context(step):
            return None

        try:
            # Build the query for this step
            query = ContextStrategy.build_context_query(
                step=step,
                user_request=user_request,
                state=self.state
            )

            # Get collection name
            collection_name = ContextStrategy.get_collection_name(step)

            # Retrieve context
            top_k = step.get('vector_top_k', 5)
            max_length = step.get('vector_max_context_length', None)

            context = self.context_manager.get_relevant_context(
                query=query,
                collection_name=collection_name,
                top_k=top_k,
                max_context_length=max_length
            )

            if context:
                logger.info(f"✓ Retrieved context from collection '{collection_name}'")
            else:
                logger.debug(f"No context found in collection '{collection_name}'")

            return context

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"⚠ Context retrieval connection error: {e}")
            if self.strict_mode:
                raise
            return None
            
        except (ValueError, KeyError) as e:
            logger.error(f"⚠ Context retrieval configuration error: {e}")
            if self.strict_mode:
                raise
            return None
            
        except NotImplementedError as e:
            logger.warning(f"⚠ Context retrieval not implemented: {e}")
            if self.strict_mode:
                raise
            return None
            
        except Exception as e:
            logger.error(f"⚠ Unexpected context retrieval error: {e}")
            if self.strict_mode:
                raise
            return None

    # -- Template resolution --------------------------------------------------

    def _resolve(self, value: str) -> str:
        """Resolve {key} placeholders in a string using current state."""
        if not isinstance(value, str):
            return value
        try:
            return value.format(**self.state)
        except (KeyError, ValueError, IndexError):
            return value

    # -- Step dispatcher -------------------------------------------------------

    def execute_step(self, step: Dict[str, Any]) -> str:
        """
        Execute a single workflow step.

        Dispatches to the appropriate handler based on the step's ``type``
        field.  Steps without an explicit type default to ``agent`` for
        backward compatibility.
        """
        step_type = step.get('type', 'agent')

        if step_type == 'file_read':
            return self._execute_file_read(step)
        elif step_type == 'file_write':
            return self._execute_file_write(step)
        elif step_type == 'status_update':
            return self._execute_status_update(step)
        elif step_type == 'vector_embed':
            return self._execute_vector_embed(step)
        else:
            return self._execute_agent_step(step)

    # -- Agent step (original behaviour) ---------------------------------------

    def _execute_agent_step(self, step: Dict[str, Any]) -> str:
        """Execute an LLM agent step (the default step type)."""
        agent_name = self._resolve(step['agent'])
        model = step['model']
        input_source = step.get('input_source', 'user_request')
        output_key = step['output_key']

        # Load system prompt
        system_prompt = self.load_system_prompt(agent_name)

        # Get input
        if input_source == 'user_request':
            user_input = self.state.get('user_request', '')
        else:
            user_input = self.state.get(input_source, '')

        # Add context from previous steps if specified
        context_keys = step.get('context_keys', [])
        if context_keys:
            context_parts = []
            for key in context_keys:
                if key in self.state and self.state[key]:
                    context_parts.append(f"## {key.replace('_', ' ').title()}\n{self.state[key]}")

            if context_parts:
                user_input = "\n\n".join([
                    "# Context from Previous Steps",
                    "\n\n".join(context_parts),
                    "# Current Task",
                    user_input
                ])

        # Retrieve and inject vector database context
        vector_context = self.retrieve_vector_context(
            step=step,
            user_request=self.state.get('user_request', '')
        )

        if vector_context:
            # Inject vector context before the current task
            user_input = "\n\n".join([
                vector_context,
                "---",
                "# Current Task",
                user_input
            ])

        # Execute agent
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing: {step.get('name', agent_name)}")
        logger.info(f"Agent: {agent_name} | Model: {model}")
        logger.info(f"{'='*60}")

        response = self.client.generate(
            model=model,
            system_prompt=system_prompt,
            user_input=user_input
        )

        # Store result in state
        self.state[output_key] = response

        return response

    # -- File read step --------------------------------------------------------

    def _execute_file_read(self, step: Dict[str, Any]) -> str:
        """Read a file and store its content in state."""
        file_path = Path(self._resolve(step['file_path']))
        output_key = step['output_key']

        if file_path.exists():
            content = file_path.read_text()
            logger.info(f"Read file: {file_path}")
        else:
            content = step.get('default', '')
            logger.info(f"File not found, using default: {file_path}")

        self.state[output_key] = content
        return content

    # -- File write step -------------------------------------------------------

    def _execute_file_write(self, step: Dict[str, Any]) -> str:
        """Write content from state to a file."""
        file_path = Path(self._resolve(step['file_path']))
        input_source = step['input_source']
        content = self.state.get(input_source, '')

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        logger.info(f"Wrote file: {file_path}")

        return content

    # -- Status update step ----------------------------------------------------

    def _execute_status_update(self, step: Dict[str, Any]) -> str:
        """Create / update a project STATUS.md from flow state (no LLM)."""
        project_dir = Path(self._resolve(step['project_dir']))
        project_dir.mkdir(parents=True, exist_ok=True)
        status_path = project_dir / 'STATUS.md'

        # Preserve key terminology from existing STATUS.md
        existing_terminology = ''
        if status_path.exists():
            existing = status_path.read_text()
            if '## Key Terminology' in existing:
                after = existing.split('## Key Terminology', 1)[1]
                term_lines = []
                for line in after.split('\n')[1:]:
                    if line.startswith('## '):
                        break
                    term_lines.append(line)
                existing_terminology = '\n'.join(term_lines).strip()

        # Collect files in project dir (excluding STATUS.md itself)
        files = sorted(
            f.name for f in project_dir.iterdir()
            if f.is_file() and f.name != 'STATUS.md'
        ) if project_dir.exists() else []
        file_list = '\n'.join(f'- {f}' for f in files) if files else '(none yet)'

        phase = self.state.get('phase', 'unknown')
        focus = self.state.get('user_request', '')
        model = os.getenv('REASONING_MODEL', 'default')
        embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

        status_content = (
            f"---\n"
            f"model: {model}\n"
            f"embedding_model: {embedding_model}\n"
            f"---\n\n"
            f"## Phase\n{phase}\n\n"
            f"## Files\n{file_list}\n\n"
            f"## Current Focus\n{focus}\n\n"
            f"## Key Terminology\n{existing_terminology}\n"
        )

        status_path.write_text(status_content)
        self.state['status'] = status_content
        logger.info(f"Updated STATUS.md in {project_dir}")
        return status_content

    # -- Vector embed step -----------------------------------------------------

    def _execute_vector_embed(self, step: Dict[str, Any]) -> str:
        """Embed content into a vector collection for later retrieval."""
        if not self.vector_context_enabled or not self.context_manager:
            logger.info("Vector context disabled, skipping embed")
            return ''

        collection_name = self._resolve(step.get('vector_collection', 'default'))
        input_source = step['input_source']
        content = self.state.get(input_source, '')

        if not content:
            logger.info("No content to embed")
            return ''

        try:
            collection = self.context_manager.get_or_create_collection(collection_name)
            doc_id = f"{self.state.get('project', 'unknown')}_{self.state.get('phase', 'unknown')}"
            collection.upsert(
                documents=[content],
                ids=[doc_id],
                metadatas=[{
                    "project": self.state.get('project', ''),
                    "phase": self.state.get('phase', ''),
                }],
            )
            logger.info(f"Embedded content into collection '{collection_name}'")
        except Exception as e:
            logger.warning(f"Failed to embed content: {e}")

        return content

    def run_flow(
        self,
        flow_name: str,
        user_request: str,
        verbose: bool = True,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete workflow.

        Args:
            flow_name: Name of the flow YAML file (without extension)
            user_request: The initial user request/prompt
            verbose: Whether to print intermediate results
            params: Optional extra parameters (e.g. project, phase) merged
                    into the initial state so that YAML templates can
                    reference them as ``{project}`` / ``{phase}``.

        Returns:
            Final state dictionary containing all step outputs

        Raises:
            FileNotFoundError: If flow or prompts don't exist
            Various exceptions from step execution
        """
        # Reset state
        self.state = {'user_request': user_request}
        if params:
            self.state.update(params)

        # Load flow configuration
        flow_config = self.load_flow(flow_name)

        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting Flow: {flow_config.get('name', flow_name)}")
        logger.info(f"Description: {flow_config.get('description', 'N/A')}")
        logger.info(f"{'#'*60}\n")

        # Execute each step
        for step in flow_config.get('steps', []):
            try:
                result = self.execute_step(step)

                if verbose:
                    output_label = step.get('output_key', step.get('id', 'step'))
                    logger.info(f"\n{'-'*40}")
                    logger.info(f"Output ({output_label}):")
                    logger.info(f"{'-'*40}")
                    # Truncate long outputs for display
                    preview = result[:500] + "..." if len(result) > 500 else result
                    logger.info(preview)

            except Exception as e:
                logger.error(f"ERROR in step {step.get('name', step.get('id', 'unknown'))}: {e}")
                raise

        logger.info(f"\n{'#'*60}")
        logger.info("Flow Complete!")
        logger.info(f"{'#'*60}")

        return self.state

    def get_available_flows(self) -> List[str]:
        """
        List all available flow configurations.
        
        Returns:
            List of flow names (without .yaml extension)
        """
        flows = []
        for flow_file in self.flows_dir.glob("*.yaml"):
            flows.append(flow_file.stem)
        return flows

    def get_context_manager(self) -> Optional[ContextManager]:
        """
        Get the context manager instance.

        Returns:
            ContextManager instance or None if disabled
        """
        return self.context_manager