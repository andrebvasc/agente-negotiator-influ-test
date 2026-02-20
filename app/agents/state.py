"""LangGraph state definition for negotiator agent."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class NegotiatorState(TypedDict):
    thread_id: str
    influencer_phone: str
    agent_id: str
    platform: Optional[str]
    deliverable_type: Optional[str]
    niche: Optional[str]
    qty: Optional[int]
    avg_views: Optional[int]
    deadline: Optional[str]
    target_cpm_brl: Optional[float]
    suggested_range: Optional[dict]  # {floor, target, ceiling}
    benchmarks: Optional[dict]  # stats from retrieve_benchmarks
    approval_required: bool
    current_offer_brl: Optional[float]
    owner: str  # "agent" | "human"
    last_user_message: str
    current_node: str
    qualification_complete: bool
    messages: Annotated[list, add_messages]  # append-only
