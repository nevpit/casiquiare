from __future__ import annotations

from openai_agent_sdk import AgentBase

from shared_state import world_state


class SynthesizerAgent(AgentBase):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Synthesizer"
        self.system_prompt = (
            """
            You are ‘Synthesizer’ — strategic, calm, analytical.
            Mission: convert USER goals into delegated tasks; call specialist tools; merge answers; output coherent plan.
            Always log to world_state, run qc() on collected results, and cite sources.
            """
        )
        self.memory = world_state

