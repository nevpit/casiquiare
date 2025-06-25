"""Brain agent for data engineering and ML modeling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Sequence, get_origin
import json

from log_config import setup_logger

from . import brain_tools
from .brain_outputs import (
    ModelResult,
    PredictionResult,
    ClusterResult,
    ExecutionResult,
)
from world_state import world_state, log_message

try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    openai = client
except Exception:  # pragma: no cover - library may be missing
    client = None
    openai = None


logger = setup_logger("casiquiare.brain")


@dataclass
class Brain:
    """Data engineer and ML modeller agent."""

    model: str = "gpt-4-turbo"
    tools: Dict[str, Any] = field(init=False, default_factory=dict)
    system_prompt: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if openai is not None:
            self.tools = brain_tools.TOOLS
            self.system_prompt = (
                "You are the Brain agent, a data engineer & ML modeler specializing in "
                "archaeological site prediction. You rigorously analyze data, run code, "
                "and return factual, reproducible results. Use tools for modeling, "
                "statistics, and clustering – avoid speculation. Always respond with JSON "
                "only, formatted as {\"agent\": \"brain\", \"type\": <type>, \"content\": <data>} "
                "and wrap it in a single markdown block."
            )

    def set_system_prompt(self, prompt: str) -> None:
        """Override the default system prompt."""
        self.system_prompt = prompt

    def ask(self, query: str) -> str:
        """Query the language model using the Brain persona."""
        if openai is None:
            raise RuntimeError("OpenAI SDK is not available.")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(model=self.model, messages=messages)
        reply = response.choices[0].message.content.strip()
        log_message("brain", "chat", reply)
        return reply

    # ------------------------------------------------------------------
    # Lightweight planning via function calling

    def _tool_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI tool specs for the registered tools."""
        import inspect

        specs: list[dict[str, Any]] = []
        for name, func in self.tools.items():
            doc_lines = (func.__doc__ or "").strip().splitlines()
            doc = doc_lines[0] if doc_lines else ""
            params: dict[str, Any] = {}
            required: list[str] = []
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
                            "integer"
                            if ann is int
                            else ("number" if ann is float else "boolean")
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

    def use_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a registered tool and log the action and result."""
        from world_state import set_value

        if name not in self.tools:
            raise KeyError(f"Tool {name} not registered")
        log_message("brain", "delegate", {"tool": name})
        result = self.tools[name](*args, **kwargs)
        set_value(f"brain_{name}", result)
        log_message("brain", "result", {"tool": name})
        return result

    def plan_and_act(self, goal: str, *, max_steps: int = 2) -> str:
        """Autonomously decide to run tools such as ``exec_code``."""

        if client is None:
            raise RuntimeError("OpenAI SDK is not available.")

        from world_state import set_value

        log_message("brain", "plan_start", {"goal": goal})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": goal},
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
            set_value("brain_answer", answer)
            log_message("brain", "conclusion", answer)
            log_message("brain", "final_output", answer)
            return answer

        return ""

    # ------------------------------------------------------------------
    # Wrappers around tool functions that also update the world_state

    def ingest_training_data(
        self, csv_path: Optional[str] = None, mapping_file: Optional[str] = None
    ) -> "pd.DataFrame":
        """Delegate to :func:`agents.brain.brain_tools.ingest_training_data`."""
        return brain_tools.ingest_training_data(csv_path, mapping_file)

    def train_model(
        self,
        data: Optional["pd.DataFrame"] = None,
        *,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
        model_path: str = "brain_model.joblib",
        model_type: str = "random_forest",
    ) -> ModelResult:
        """Train a model and store the summary in ``world_state``.

        Parameters mirror :func:`agents.brain.brain_tools.train_model`.
        """
        result = brain_tools.train_model(
            data,
            n_estimators=n_estimators,
            random_state=random_state,
            model_path=model_path,
            model_type=model_type,
        )
        info = ModelResult(
            model_type=result["model_type"],
            metrics=result["metrics"],
            feature_importances=result["feature_importances"],
            model_path=result["model_path"],
            model=result["model"],
        )
        world_state["latest_model"] = asdict(info)
        logger.info("Stored latest_model in world_state")
        logger.debug("Model metrics: %s", info.metrics)
        log_message("brain", "train_model", asdict(info))
        return info

    def load_model(self, model_path: str) -> ModelResult:
        """Reload a model from disk and store it in ``world_state``."""
        model = brain_tools.load_model(model_path)
        importances = (
            {str(i): float(v) for i, v in enumerate(model.feature_importances_)}
            if hasattr(model, "feature_importances_")
            else {}
        )
        info = ModelResult(
            model_type=type(model).__name__,
            metrics={},
            feature_importances=importances,
            model_path=model_path,
            model=model,
        )
        world_state["latest_model"] = asdict(info)
        logger.info("Loaded model from %s", model_path)
        log_message("brain", "load_model", asdict(info))
        return info

    def score_likelihood(self, model: Any, data: Sequence[Sequence[float]]):
        """Score data and record the raw scores."""
        scores = brain_tools.score_likelihood(model, data)
        world_state["latest_prediction"] = scores
        logger.info("Stored latest_prediction scores in world_state")
        log_message("brain", "score_likelihood", scores)
        return scores

    def predict_sites(
        self,
        model: Any,
        input_data: Any,
        *,
        grid_size: float = 0.01,
        top_n: int = 5,
    ) -> PredictionResult:
        """Predict site likelihoods and update ``world_state``."""
        result = brain_tools.predict_sites(
            model,
            input_data,
            grid_size=grid_size,
            top_n=top_n,
        )
        pred = PredictionResult(
            scores_map=result["scores_map"],
            summary=result["summary"],
            feature_importances=result.get("feature_importances"),
        )
        world_state["latest_prediction"] = asdict(pred)
        logger.info("Stored latest_prediction map in world_state")
        log_message("brain", "predict_sites", asdict(pred))
        return pred

    def validate_features(self, features: Sequence[Dict[str, Any]]):
        """Validate features and store the results."""
        res = brain_tools.validate_features(features)
        world_state["validation"] = res
        logger.info("Stored validation results in world_state")
        log_message("brain", "validate_features", res)
        return res

    def cluster_features(
        self,
        features: Sequence[Any],
        *,
        eps: float = 100.0,
        min_samples: int = 2,
        area_id: Optional[str] = None,
    ) -> ClusterResult:
        """Cluster geographic features and record the cluster summary."""
        result = brain_tools.cluster_features(features, eps=eps, min_samples=min_samples)
        clusters = world_state.setdefault("clusters", {})
        cluster = ClusterResult(labels=result["labels"], summary=result["summary"])
        if area_id is not None:
            clusters[area_id] = asdict(cluster)
        else:
            clusters["latest"] = asdict(cluster)
        world_state["clusters"] = clusters
        logger.info("Stored clusters summary in world_state")
        log_message("brain", "cluster_features", asdict(cluster))
        return cluster

    def exec_code(
        self,
        code: str,
        local_vars: Optional[Dict[str, Any]] = None,
        *,
        timeout: int = 5,
    ) -> ExecutionResult:
        """Execute code via :func:`agents.brain.brain_tools.exec_code`."""
        result = brain_tools.exec_code(code, local_vars, timeout=timeout)
        exec_res = ExecutionResult(
            stdout=result.get("stdout", ""),
            result=result.get("result"),
            figures=result.get("figures", []),
            locals=result.get("locals"),
            error=result.get("error"),
        )
        world_state["last_exec"] = asdict(exec_res)
        logger.info("Stored last_exec result in world_state")
        log_message("brain", "exec_code", asdict(exec_res))
        return exec_res

    def get_results(self, key: Optional[str] = None) -> Any:
        """Return stored results from ``world_state``."""
        if key is None:
            logger.info("Retrieving full world_state")
            return world_state
        logger.info("Retrieving %s from world_state", key)
        return world_state.get(key)


TOOLS: Dict[str, Any] = brain_tools.TOOLS

__all__ = ["Brain", "TOOLS"]
