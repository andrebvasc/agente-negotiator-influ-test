"""SQLAlchemy 2.0 declarative models."""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False, default="")
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="agent")


class Influencer(Base):
    __tablename__ = "influencers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phone: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    niche: Mapped[str | None] = mapped_column(String(64), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(32), nullable=True)
    avg_views: Mapped[int | None] = mapped_column(Integer, nullable=True)

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="influencer"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    influencer_id: Mapped[int] = mapped_column(
        ForeignKey("influencers.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active"
    )
    owner: Mapped[str] = mapped_column(String(16), nullable=False, default="agent")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    agent: Mapped["Agent"] = relationship(back_populates="conversations")
    influencer: Mapped["Influencer"] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")
    offers: Mapped[list["Offer"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Deal(Base):
    __tablename__ = "deals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    influencer_name: Mapped[str] = mapped_column(String(128), nullable=False)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    niche: Mapped[str] = mapped_column(String(64), nullable=False)
    deliverable_type: Mapped[str] = mapped_column(String(32), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    avg_views: Mapped[int] = mapped_column(Integer, nullable=False)
    final_price_brl: Mapped[float] = mapped_column(Float, nullable=False)
    cpm_brl: Mapped[float] = mapped_column(Float, nullable=False)
    closed_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Offer(Base):
    __tablename__ = "offers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id"), nullable=False
    )
    floor_brl: Mapped[float] = mapped_column(Float, nullable=False)
    target_brl: Mapped[float] = mapped_column(Float, nullable=False)
    ceiling_brl: Mapped[float] = mapped_column(Float, nullable=False)
    proposed_brl: Mapped[float | None] = mapped_column(Float, nullable=True)
    accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="offers")
