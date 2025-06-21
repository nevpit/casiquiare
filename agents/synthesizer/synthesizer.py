"""Synthesizer agent orchestrating other agents."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, get_origin
import inspect
import json

from log_config import setup_logger
from agents.eyes import Eyes, eyes_tools as eyes_tools
from agents.brain import Brain
from agents.memory import MemoryKeeper
from agents.context import ContextEngine
from world_state import world_state, set_value, get_value, log_message

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:  # pragma: no cover - library may be missing
    client = None

logger = setup_logger("casiquiare.synthesizer")


@dataclass
class Synthesizer:
    """Integrative archaeologist and project orchestrator."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")
    eyes: Eyes = field(init=False)
    brain: Brain = field(init=False)
    memory_keeper: MemoryKeeper = field(init=False)
    context_engine: ContextEngine = field(init=False)
    world_state: Dict[str, Any] = field(init=False, default_factory=lambda: world_state)

    def __post_init__(self) -> None:
        """Instantiate subordinate agents and register their tools."""

        # Create subordinate agent instances for orchestrated access
        self.eyes = Eyes()
        self.brain = Brain()
        self.memory_keeper = MemoryKeeper()
        self.context_engine = ContextEngine()

        if client is not None:
            self.system_prompt = (
                "You are the Synthesizer, an Integrative Archaeologist and Project "
                "Orchestrator. You steer research goals and coordinate the Eyes, Brain, "
                "Memory Keeper, and Context Engine agents. Provide decisive, synthesis-"
                "focused guidance. Always present your response as a JSON object in the "
                "form {\"agent\": \"synthesizer\", \"type\": <type>, \"content\": <data>} "
                "inside a single markdown block. After the JSON, give a brief executive "
                "summary of 2–3 sentences."
            )

        # Register a selection of sub-agent capabilities as callable tools
        self.tools = {
            # Eyes tools
            "eyes_analyze_lidar": self.eyes.analyze_lidar,
            "eyes_scan_area": eyes_tools.scan_area,
            "eyes_analyze_satellite": self.eyes.analyze_satellite_image,
            "eyes_lidar_feature_detection": self.eyes.lidar_feature_detection,
            # Brain tools
            "brain_train_model": self.brain.train_model,
            "brain_score_likelihood": self.brain.score_likelihood,
            "brain_validate_features": self.brain.validate_features,
            "brain_cluster_features": self.brain.cluster_features,
            "brain_exec_code": self.brain.exec_code,
            # Memory Keeper tools
            "memory_search_corpus": self.memory_keeper.search_corpus,
            "memory_semantic_search": self.memory_keeper.semantic_search,
            "memory_geocode_place": self.memory_keeper.geocode_place,
            "memory_extract_locations": self.memory_keeper.extract_locations,
            # Context Engine tools
            "context_analyze_environment": self.context_engine.analyze_environment,
            "context_generate_summary": self.context_engine.generate_context_summary,
        }

        logger.info("Synthesizer initialized with tools: %s", list(self.tools))

    # ------------------------------------------------------------------
    # Shared memory helpers

    def gather_context(self) -> Dict[str, Any]:
        """Return relevant entries from ``world_state`` for planning."""
        context_keys = ["latest_model", "latest_prediction", "clusters"]
        return {k: get_value(k) for k in context_keys if get_value(k) is not None}

    def use_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered tool and log the action and result."""
        if name not in self.tools:
            raise KeyError(f"Tool {name} not registered")
        log_message("synthesizer", "delegate", {"tool": name})
        result = self.tools[name](*args, **kwargs)
        set_value(f"synth_{name}", result)
        log_message("synthesizer", "result", {"tool": name})
        return result

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Synthesizer persona."""
        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content.strip()
        log_message("synthesizer", "chat", reply)
        return reply

    # ------------------------------------------------------------------
    # LLM planning with tool use

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """Return OpenAI function specs for registered tools."""
        specs: List[Dict[str, Any]] = []
        for name, func in self.tools.items():
            doc_lines = (func.__doc__ or "").strip().splitlines()
            doc = doc_lines[0] if doc_lines else ""
            params: Dict[str, Any] = {}
            required: List[str] = []
            try:
                sig = inspect.signature(func)
                for p_name, param in sig.parameters.items():
                    p_type = "string"
                    ann = param.annotation
                    origin = get_origin(ann)
                    if origin is not None:
                        ann = origin
                    if ann in (int, float, bool):
                        p_type = (
                            "integer" if ann is int else ("number" if ann is float else "boolean")
                        )
                    elif ann is list:
                        p_type = "array"
                    elif ann is dict:
                        p_type = "object"
                    params[p_name] = {"type": p_type}
                    if param.default is inspect._empty:
                        required.append(p_name)
            except (TypeError, ValueError):
                pass
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": doc,
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": required,
                        },
                    },
                }
            )
        return specs

    def plan_and_act(self, goal: str, *, max_steps: int = 6) -> str:
        """Use LLM function calling to plan and execute tools."""

        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")

        log_message("synthesizer", "plan_start", {"goal": goal})

        context = self.gather_context()
        user_content = goal
        if context:
            user_content += "\nContext:\n" + json.dumps(context)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = self._tool_specs()

        step = 0
        while step < max_steps:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            step += 1

            if getattr(msg, "tool_calls", None):
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for call in msg.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except Exception:
                        args = {}
                    try:
                        result = self.use_tool(name, **args)
                    except Exception as exc:  # pragma: no cover - runtime safety
                        result = {"error": str(exc)}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
                continue

            answer = msg.content or ""
            set_value("synth_answer", answer)
            log_message("synthesizer", "conclusion", answer)
            log_message("synthesizer", "final_output", answer)
            return answer

        timeout = json.dumps({"status": "incomplete", "reason": "max_steps_exceeded"})
        set_value("synth_answer", timeout)
        log_message("synthesizer", "conclusion", timeout)
        log_message("synthesizer", "final_output", timeout)
        return timeout


__all__ = ["Synthesizer"]

