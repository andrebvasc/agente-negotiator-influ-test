"""Definição do estado LangGraph para o agente negociador."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class NegotiatorState(TypedDict):
    thread_id: str
    influencer_phone: str
    agent_id: str
    name: Optional[str]
    platform: Optional[str]
    deliverable_type: Optional[str]
    niche: Optional[str]
    qty: Optional[int]
    avg_views: Optional[int]
    deadline: Optional[str]
    target_cpm_brl: Optional[float]
    suggested_range: Optional[dict]  # {floor, target, ceiling}
    benchmarks: Optional[dict]  # estatísticas de retrieve_benchmarks
    approval_required: bool
    current_offer_brl: Optional[float]
    owner: str  # "agent" | "human"
    last_user_message: str
    current_node: str
    qualification_complete: bool
    messages: Annotated[list, add_messages]  # somente adição
    conversation_history: Optional[list]  # [{"role": "user"/"assistant", "content": "..."}]
    influencer_id: Optional[int]
    influencer_updates: Optional[dict]  # campos para persistir na tabela influencers
    last_agent_offer_brl: Optional[float]  # último preço proposto pelo agente
    deal_accepted: bool
    agreed_price_brl: Optional[float]
    deal_to_save: Optional[dict | list]  # dict único ou lista de dicts (multi-plataforma)
    operator_counter_offer_brl: Optional[float]  # contraproposta definida pelo operador
    platform_details: Optional[dict]  # {"instagram": {"qty": 5, "avg_views": 100000}, ...}
    benchmarks_per_platform: Optional[dict]
    suggested_range_per_platform: Optional[dict]
