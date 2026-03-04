"""Operações CRUD para dados de negócio."""

import uuid

from sqlalchemy.orm import Session

from app.db.models import Agent, Conversation, Deal, Influencer, Message, Offer


def get_conversation_messages(
    session: Session, conversation_id: int, limit: int = 20
) -> list[dict]:
    """Retorna as últimas N mensagens de uma conversa como dicts."""
    msgs = (
        session.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    msgs = msgs[-limit:] if len(msgs) > limit else msgs
    return [{"role": m.role, "content": m.content} for m in msgs]


def update_influencer_profile(session: Session, influencer_id: int, **kwargs) -> None:
    """Atualiza campos do influenciador que ainda são NULL (nunca sobrescreve dados existentes).

    Tratamento especial para ``platform``: faz merge dos novos valores com os existentes
    (set separado por vírgula).
    """
    influencer = session.query(Influencer).get(influencer_id)
    if not influencer:
        return
    for key, value in kwargs.items():
        if not value or not hasattr(influencer, key):
            continue
        if key == "platform":
            existing = set(influencer.platform.split(",")) if influencer.platform else set()
            existing.discard("")
            new_vals = set(value.split(",")) if isinstance(value, str) else set(value)
            merged = existing | new_vals
            influencer.platform = ",".join(sorted(merged))
        elif not getattr(influencer, key):
            setattr(influencer, key, value)
    session.flush()


def get_or_create_agent(session: Session, agent_id: str, name: str) -> Agent:
    agent = session.query(Agent).filter_by(agent_id=agent_id).first()
    if not agent:
        agent = Agent(agent_id=agent_id, name=name, persona="")
        session.add(agent)
        session.flush()
    return agent


def get_or_create_influencer(session: Session, phone: str) -> Influencer:
    influencer = session.query(Influencer).filter_by(phone=phone).first()
    if not influencer:
        influencer = Influencer(phone=phone)
        session.add(influencer)
        session.flush()
    return influencer


def create_conversation(
    session: Session, agent: Agent, influencer: Influencer
) -> Conversation:
    thread_id = str(uuid.uuid4())
    conv = Conversation(
        thread_id=thread_id,
        agent_id=agent.id,
        influencer_id=influencer.id,
        status="active",
        owner="agent",
    )
    session.add(conv)
    session.flush()
    return conv


def get_active_conversation(
    session: Session, agent_id: int, influencer_id: int
) -> Conversation | None:
    return (
        session.query(Conversation)
        .filter_by(agent_id=agent_id, influencer_id=influencer_id, status="active")
        .first()
    )


def save_message(
    session: Session, conversation_id: int, role: str, content: str
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    session.flush()
    return msg


def save_offer(
    session: Session,
    conversation_id: int,
    floor_brl: float,
    target_brl: float,
    ceiling_brl: float,
    proposed_brl: float | None = None,
    accepted: bool | None = None,
) -> Offer:
    offer = Offer(
        conversation_id=conversation_id,
        floor_brl=floor_brl,
        target_brl=target_brl,
        ceiling_brl=ceiling_brl,
        proposed_brl=proposed_brl,
        accepted=accepted,
    )
    session.add(offer)
    session.flush()
    return offer


def list_conversations(session: Session) -> list[Conversation]:
    return session.query(Conversation).order_by(Conversation.created_at.desc()).all()


def update_conversation_owner(
    session: Session, conversation_id: int, owner: str
) -> None:
    conv = session.query(Conversation).get(conversation_id)
    if conv:
        conv.owner = owner
        session.flush()


def save_deal(session: Session, deal_data: dict) -> Deal:
    """Cria um registro Deal a partir do dict deal_to_save."""
    deal = Deal(
        influencer_name=deal_data.get("influencer_name", ""),
        influencer_phone=deal_data.get("influencer_phone"),
        platform=deal_data.get("platform", ""),
        niche=deal_data.get("niche", ""),
        deliverable_type=deal_data.get("deliverable_type", ""),
        qty=deal_data.get("qty", 1),
        avg_views=deal_data.get("avg_views", 0),
        final_price_brl=deal_data.get("final_price_brl", 0.0),
        cpm_brl=deal_data.get("cpm_brl", 0.0),
    )
    session.add(deal)
    session.flush()
    return deal


def update_conversation_status(
    session: Session, conversation_id: int, status: str
) -> None:
    conv = session.query(Conversation).get(conversation_id)
    if conv:
        conv.status = status
        session.flush()
