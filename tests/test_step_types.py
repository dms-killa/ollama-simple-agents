"""Tests for the new step types (file_read, file_write, status_update, vector_embed)
and the thought_piece_workflow plumbing.

Ollama is mocked throughout. Chroma uses a temp directory.
"""

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from core.flow_engine import FlowEngine
from core.ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """OllamaClient that returns a canned response."""
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = "LLM response placeholder"
    client.resolve_model.side_effect = lambda m: m
    return client


@pytest.fixture
def engine(mock_client, tmp_path):
    """FlowEngine wired to the mock client, vector context disabled."""
    return FlowEngine(
        client=mock_client,
        flows_dir=str(tmp_path / "flows"),
        prompts_dir="prompts",
        enable_vector_context=False,
    )


@pytest.fixture
def engine_with_state(engine):
    """Engine with basic state pre-populated."""
    engine.state = {
        "user_request": "Write about testing",
        "project": "test-proj",
        "phase": "vision",
    }
    return engine


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------

class TestResolveTemplate:
    def test_resolves_known_keys(self, engine_with_state):
        assert engine_with_state._resolve("projects/{project}/{phase}.md") == "projects/test-proj/vision.md"

    def test_leaves_unknown_keys(self, engine_with_state):
        assert engine_with_state._resolve("{missing}") == "{missing}"

    def test_non_string_passthrough(self, engine_with_state):
        assert engine_with_state._resolve(42) == 42


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------

class TestFileRead:
    def test_reads_existing_file(self, engine_with_state, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("hello world")
        step = {"file_path": str(p), "output_key": "content"}
        result = engine_with_state._execute_file_read(step)
        assert result == "hello world"
        assert engine_with_state.state["content"] == "hello world"

    def test_missing_file_returns_default(self, engine_with_state, tmp_path):
        step = {
            "file_path": str(tmp_path / "nope.txt"),
            "output_key": "content",
            "default": "fallback",
        }
        result = engine_with_state._execute_file_read(step)
        assert result == "fallback"
        assert engine_with_state.state["content"] == "fallback"

    def test_missing_file_empty_default(self, engine_with_state, tmp_path):
        step = {"file_path": str(tmp_path / "nope.txt"), "output_key": "content"}
        result = engine_with_state._execute_file_read(step)
        assert result == ""

    def test_template_resolution_in_path(self, engine_with_state, tmp_path):
        proj_dir = tmp_path / "projects" / "test-proj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "STATUS.md").write_text("status data")
        step = {
            "file_path": str(tmp_path / "projects/{project}/STATUS.md"),
            "output_key": "status",
        }
        engine_with_state.state["project"] = "test-proj"
        result = engine_with_state._execute_file_read(step)
        assert result == "status data"


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------

class TestFileWrite:
    def test_writes_file(self, engine_with_state, tmp_path):
        engine_with_state.state["draft"] = "my draft content"
        out = tmp_path / "out" / "draft.md"
        step = {"file_path": str(out), "input_source": "draft"}
        result = engine_with_state._execute_file_write(step)
        assert out.read_text() == "my draft content"
        assert result == "my draft content"

    def test_creates_parent_dirs(self, engine_with_state, tmp_path):
        engine_with_state.state["draft"] = "content"
        out = tmp_path / "a" / "b" / "c" / "file.md"
        step = {"file_path": str(out), "input_source": "draft"}
        engine_with_state._execute_file_write(step)
        assert out.exists()

    def test_template_resolution_in_path(self, engine_with_state, tmp_path):
        engine_with_state.state["draft"] = "vision text"
        step = {
            "file_path": str(tmp_path / "projects/{project}/{phase}.md"),
            "input_source": "draft",
        }
        engine_with_state._execute_file_write(step)
        written = (tmp_path / "projects" / "test-proj" / "vision.md").read_text()
        assert written == "vision text"


# ---------------------------------------------------------------------------
# status_update
# ---------------------------------------------------------------------------

class TestStatusUpdate:
    def test_creates_status_md(self, engine_with_state, tmp_path):
        proj_dir = tmp_path / "projects" / "test-proj"
        step = {"project_dir": str(proj_dir)}
        result = engine_with_state._execute_status_update(step)

        status_path = proj_dir / "STATUS.md"
        assert status_path.exists()
        content = status_path.read_text()
        assert "## Phase" in content
        assert "vision" in content
        assert "## Current Focus" in content
        assert "Write about testing" in content
        assert "## Key Terminology" in content

    def test_has_yaml_frontmatter(self, engine_with_state, tmp_path):
        proj_dir = tmp_path / "myproj"
        step = {"project_dir": str(proj_dir)}
        engine_with_state._execute_status_update(step)
        content = (proj_dir / "STATUS.md").read_text()
        assert content.startswith("---\n")
        assert "model:" in content
        assert "embedding_model:" in content

    def test_lists_existing_files(self, engine_with_state, tmp_path):
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "vision.md").write_text("v")
        (proj_dir / "outline.md").write_text("o")
        step = {"project_dir": str(proj_dir)}
        engine_with_state._execute_status_update(step)
        content = (proj_dir / "STATUS.md").read_text()
        assert "- outline.md" in content
        assert "- vision.md" in content

    def test_preserves_terminology(self, engine_with_state, tmp_path):
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir(parents=True)
        existing = textwrap.dedent("""\
            ---
            model: old
            embedding_model: old
            ---

            ## Phase
            outline

            ## Files
            (none yet)

            ## Current Focus
            old focus

            ## Key Terminology
            - Synth-wave: a genre of electronic music
            - Retrofuturism: imagining the future from a past perspective
        """)
        (proj_dir / "STATUS.md").write_text(existing)
        step = {"project_dir": str(proj_dir)}
        engine_with_state._execute_status_update(step)
        content = (proj_dir / "STATUS.md").read_text()
        assert "Synth-wave" in content
        assert "Retrofuturism" in content
        # Phase should be updated to current
        assert "vision" in content

    def test_template_resolution_in_project_dir(self, engine_with_state, tmp_path):
        step = {"project_dir": str(tmp_path / "projects/{project}")}
        engine_with_state._execute_status_update(step)
        assert (tmp_path / "projects" / "test-proj" / "STATUS.md").exists()


# ---------------------------------------------------------------------------
# vector_embed
# ---------------------------------------------------------------------------

class TestVectorEmbed:
    def test_skips_when_disabled(self, engine_with_state):
        step = {"vector_collection": "col", "input_source": "draft"}
        engine_with_state.state["draft"] = "content"
        result = engine_with_state._execute_vector_embed(step)
        assert result == ""

    def test_skips_empty_content(self, engine_with_state):
        engine_with_state.vector_context_enabled = True
        engine_with_state.context_manager = MagicMock()
        step = {"vector_collection": "col", "input_source": "draft"}
        # draft not in state → empty
        result = engine_with_state._execute_vector_embed(step)
        assert result == ""

    def test_calls_upsert(self, engine_with_state):
        mock_cm = MagicMock()
        mock_collection = MagicMock()
        mock_cm.get_or_create_collection.return_value = mock_collection
        engine_with_state.vector_context_enabled = True
        engine_with_state.context_manager = mock_cm
        engine_with_state.state["draft"] = "some content"
        step = {"vector_collection": "myproj_research", "input_source": "draft"}
        result = engine_with_state._execute_vector_embed(step)
        assert result == "some content"
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs[1]["documents"] == ["some content"]

    def test_handles_embed_failure_gracefully(self, engine_with_state):
        mock_cm = MagicMock()
        mock_cm.get_or_create_collection.side_effect = RuntimeError("boom")
        engine_with_state.vector_context_enabled = True
        engine_with_state.context_manager = mock_cm
        engine_with_state.state["draft"] = "content"
        step = {"vector_collection": "col", "input_source": "draft"}
        # Should not raise
        result = engine_with_state._execute_vector_embed(step)
        assert result == "content"


# ---------------------------------------------------------------------------
# execute_step dispatch
# ---------------------------------------------------------------------------

class TestStepDispatch:
    def test_defaults_to_agent(self, engine_with_state, mock_client):
        step = {
            "agent": "analyst",
            "model": "REASONING_MODEL",
            "input_source": "user_request",
            "output_key": "result",
        }
        result = engine_with_state.execute_step(step)
        assert result == "LLM response placeholder"
        mock_client.generate.assert_called_once()

    def test_dispatches_file_read(self, engine_with_state, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("data")
        step = {"type": "file_read", "file_path": str(p), "output_key": "x"}
        result = engine_with_state.execute_step(step)
        assert result == "data"

    def test_dispatches_file_write(self, engine_with_state, tmp_path):
        engine_with_state.state["x"] = "written"
        step = {"type": "file_write", "file_path": str(tmp_path / "o.txt"), "input_source": "x"}
        engine_with_state.execute_step(step)
        assert (tmp_path / "o.txt").read_text() == "written"

    def test_dispatches_status_update(self, engine_with_state, tmp_path):
        step = {"type": "status_update", "project_dir": str(tmp_path / "p")}
        engine_with_state.execute_step(step)
        assert (tmp_path / "p" / "STATUS.md").exists()

    def test_dispatches_vector_embed(self, engine_with_state):
        step = {"type": "vector_embed", "vector_collection": "c", "input_source": "draft"}
        # vector disabled → returns empty, no error
        result = engine_with_state.execute_step(step)
        assert result == ""


# ---------------------------------------------------------------------------
# run_flow with params
# ---------------------------------------------------------------------------

class TestRunFlowParams:
    def test_params_merged_into_state(self, mock_client, tmp_path):
        flows_dir = tmp_path / "flows"
        flows_dir.mkdir()
        flow = {
            "name": "test",
            "steps": [
                {
                    "type": "file_read",
                    "file_path": str(tmp_path / "projects/{project}/STATUS.md"),
                    "output_key": "status",
                    "default": "none",
                }
            ],
        }
        (flows_dir / "test.yaml").write_text(yaml.dump(flow))
        engine = FlowEngine(
            client=mock_client,
            flows_dir=str(flows_dir),
            prompts_dir="prompts",
            enable_vector_context=False,
        )
        state = engine.run_flow(
            "test", "req", verbose=False, params={"project": "abc", "phase": "vision"}
        )
        assert state["project"] == "abc"
        assert state["phase"] == "vision"


# ---------------------------------------------------------------------------
# Agent step with template-resolved agent name
# ---------------------------------------------------------------------------

class TestAgentTemplateResolution:
    def test_resolves_agent_name(self, mock_client, tmp_path):
        """The {phase}_agent pattern should resolve to an actual prompt file."""
        flows_dir = tmp_path / "flows"
        flows_dir.mkdir()
        flow = {
            "name": "test",
            "steps": [
                {
                    "agent": "{phase}_agent",
                    "model": "REASONING_MODEL",
                    "input_source": "user_request",
                    "output_key": "draft",
                }
            ],
        }
        (flows_dir / "test.yaml").write_text(yaml.dump(flow))
        engine = FlowEngine(
            client=mock_client,
            flows_dir=str(flows_dir),
            prompts_dir="prompts",
            enable_vector_context=False,
        )
        state = engine.run_flow(
            "test", "Write something", verbose=False, params={"phase": "vision"}
        )
        assert state["draft"] == "LLM response placeholder"
        # Verify it actually loaded the vision_agent prompt
        mock_client.generate.assert_called_once()
        call_kwargs = mock_client.generate.call_args
        assert "vision" in call_kwargs[1]["system_prompt"].lower() or \
               "vision" in call_kwargs.kwargs.get("system_prompt", "").lower()


# ---------------------------------------------------------------------------
# Workflow YAML loads correctly
# ---------------------------------------------------------------------------

class TestThoughtPieceWorkflowConfig:
    def test_yaml_loads(self):
        flow_path = Path("config/flows/thought_piece_workflow.yaml")
        assert flow_path.exists(), "thought_piece_workflow.yaml must exist"
        with open(flow_path) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "thought_piece_workflow"
        steps = config["steps"]
        assert len(steps) == 5
        step_types = [s.get("type", "agent") for s in steps]
        assert step_types == ["file_read", "agent", "file_write", "status_update", "vector_embed"]

    def test_step_ids_unique(self):
        with open("config/flows/thought_piece_workflow.yaml") as f:
            config = yaml.safe_load(f)
        ids = [s["id"] for s in config["steps"]]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Backward compatibility: existing flows still work
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_dev_workflow_runs(self, mock_client):
        engine = FlowEngine(
            client=mock_client,
            flows_dir="config/flows",
            prompts_dir="prompts",
            enable_vector_context=False,
        )
        state = engine.run_flow("dev_workflow", "Build a calculator", verbose=False)
        assert "review" in state
        assert mock_client.generate.call_count == 4

    def test_blog_workflow_runs(self, mock_client):
        engine = FlowEngine(
            client=mock_client,
            flows_dir="config/flows",
            prompts_dir="prompts",
            enable_vector_context=False,
        )
        state = engine.run_flow("blog_workflow", "Write about AI", verbose=False)
        assert "edited_content" in state
        assert mock_client.generate.call_count == 4


# ---------------------------------------------------------------------------
# End-to-end: thought_piece_workflow with mocked Ollama
# ---------------------------------------------------------------------------

class TestThoughtPieceEndToEnd:
    # Resolve source dirs before any chdir
    _project_root = Path(__file__).resolve().parent.parent

    def test_vision_phase_creates_files(self, mock_client, tmp_path, monkeypatch):
        """Simulate: python main.py --flow thought_piece_workflow --project my-essay --phase vision --request 'Write about X'"""
        monkeypatch.chdir(tmp_path)

        # Copy prompts and flow config into the temp working dir
        import shutil
        shutil.copytree(self._project_root / "prompts", tmp_path / "prompts")
        shutil.copytree(self._project_root / "config", tmp_path / "config")

        engine = FlowEngine(
            client=mock_client,
            flows_dir="config/flows",
            prompts_dir="prompts",
            enable_vector_context=False,
        )
        state = engine.run_flow(
            "thought_piece_workflow",
            "Write about X",
            verbose=False,
            params={"project": "my-essay", "phase": "vision"},
        )

        # STATUS.md created
        status_path = tmp_path / "projects" / "my-essay" / "STATUS.md"
        assert status_path.exists()
        status = status_path.read_text()
        assert "vision" in status
        assert "Write about X" in status

        # vision.md created
        vision_path = tmp_path / "projects" / "my-essay" / "vision.md"
        assert vision_path.exists()
        assert vision_path.read_text() == "LLM response placeholder"

        # State has expected keys
        assert state["draft"] == "LLM response placeholder"
        assert "status" in state
