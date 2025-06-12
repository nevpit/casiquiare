"""Shared knowledge store between agents."""

from __future__ import annotations

from typing import Any, Dict

# Simple in-memory dictionary used by agents to exchange results.
world_state: Dict[str, Any] = {
    "latest_model": None,
    "latest_prediction": None,
    "clusters": {},
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
    world_state.update({"latest_model": None, "latest_prediction": None, "clusters": {}})


__all__ = ["world_state", "set_value", "get_value", "reset"]
