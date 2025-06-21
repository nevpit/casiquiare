"""Shared knowledge store between agents."""

from __future__ import annotations

from typing import Any, Dict, List

from dataclasses import asdict
from time import time

from agents.brain.brain_outputs import AgentMessage
from log_config import setup_logger

logger = setup_logger("casiquiare.world_state")

# Simple in-memory dictionary used by agents to exchange results.
world_state: Dict[str, Any] = {
    "latest_model": None,
    "latest_prediction": None,
    "clusters": {},
    "messages": [],
    "historical_clues": {},
}


def set_value(key: str, value: Any) -> None:
    """Store a value in the global world_state."""
    world_state[key] = value


def get_value(key: str, default: Any = None) -> Any:
    """Retrieve a value from the global world_state."""
    return world_state.get(key, default)


def reset() -> None:
    """Reset all world_state entries."""
    world_state.clear()
    world_state.update(
        {
            "latest_model": None,
            "latest_prediction": None,
            "clusters": {},
            "messages": [],
            "historical_clues": {},
        }
    )


def log_message(agent: str, msg_type: str, content: Any) -> None:
    """Append a structured message to ``world_state``."""
    msg = AgentMessage(agent=agent, type=msg_type, content=content, timestamp=time())
    msgs: List[Dict[str, Any]] = world_state.setdefault("messages", [])
    msgs.append(asdict(msg))
    world_state["messages"] = msgs
    try:
        logger.info("%s:%s - %s", agent, msg_type, content)
    except Exception:
        pass


__all__ = ["world_state", "set_value", "get_value", "reset", "log_message"]
