"""Agent registry: manages available agent configurations."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    agent_id: str
    name: str
    persona: str = ""
    config: dict = field(default_factory=dict)


class AgentRegistry:
    """Simple in-memory registry of agent configurations."""

    def __init__(self):
        self._agents: dict[str, AgentConfig] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(
            AgentConfig(
                agent_id="negotiator",
                name="Negociador de Influenciadores",
                persona=(
                    "Você é um negociador profissional e cordial que trabalha "
                    "para uma agência de marketing de influenciadores. "
                    "Negocia contratos de forma justa usando dados de mercado."
                ),
                config={"default_cpm_brl": 40.0},
            )
        )

    def register(self, config: AgentConfig) -> None:
        self._agents[config.agent_id] = config

    def get(self, agent_id: str) -> AgentConfig | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[AgentConfig]:
        return list(self._agents.values())


registry = AgentRegistry()
