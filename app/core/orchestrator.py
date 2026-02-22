"""Orchestrator: bridge between CLI and LangGraph."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.negotiator import build_graph
from app.core.registry import registry
from app.core.store import (
    create_conversation,
    get_active_conversation,
    get_conversation_messages,
    get_or_create_agent,
    get_or_create_influencer,
    save_message,
    update_conversation_owner,
    update_influencer_profile,
)
from app.db.session import SessionLocal, init_db
from app.tools.guardrails import check_human_handoff, check_sensitive_data, SENSITIVE_RESPONSE

load_dotenv()

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite")


class Orchestrator:
    """Manages conversation lifecycle between CLI and LangGraph."""

    def __init__(self, agent_id: str = "negotiator"):
        init_db()
        self.agent_config = registry.get(agent_id)
        if not self.agent_config:
            raise ValueError(f"Agent '{agent_id}' not found in registry")

        Path(CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_ctx = SqliteSaver.from_conn_string(CHECKPOINT_DB)
        self.checkpointer = self._checkpointer_ctx.__enter__()
        self.graph = build_graph(checkpointer=self.checkpointer)
        self.db_session = SessionLocal()

        self.agent = get_or_create_agent(
            self.db_session, self.agent_config.agent_id, self.agent_config.name
        )
        self.db_session.commit()

    def start_or_resume_conversation(
        self, influencer_phone: str, new: bool = False
    ) -> dict:
        """Start a new or resume existing conversation."""
        influencer = get_or_create_influencer(self.db_session, influencer_phone)
        self.db_session.commit()

        if not new:
            conv = get_active_conversation(
                self.db_session, self.agent.id, influencer.id
            )
            if conv:
                return {
                    "conversation": conv,
                    "thread_id": conv.thread_id,
                    "influencer": influencer,
                    "resumed": True,
                }

        conv = create_conversation(self.db_session, self.agent, influencer)
        self.db_session.commit()

        return {
            "conversation": conv,
            "thread_id": conv.thread_id,
            "influencer": influencer,
            "resumed": False,
        }

    def process_message(
        self,
        thread_id: str,
        conversation_id: int,
        user_message: str,
        influencer_id: int | None = None,
    ) -> dict:
        """Process a user message through the graph."""
        # Guardrails
        if check_sensitive_data(user_message):
            save_message(self.db_session, conversation_id, "user", user_message)
            save_message(
                self.db_session, conversation_id, "assistant", SENSITIVE_RESPONSE
            )
            self.db_session.commit()
            return {
                "response": SENSITIVE_RESPONSE,
                "owner": "agent",
                "approval_required": False,
            }

        handoff = check_human_handoff(user_message)

        save_message(self.db_session, conversation_id, "user", user_message)
        self.db_session.commit()

        if handoff:
            update_conversation_owner(self.db_session, conversation_id, "human")
            self.db_session.commit()
            response = (
                "Entendido! Vou transferir você para um atendente humano. "
                "Aguarde um momento, por favor."
            )
            save_message(self.db_session, conversation_id, "assistant", response)
            self.db_session.commit()
            return {
                "response": response,
                "owner": "human",
                "approval_required": False,
            }

        config = {"configurable": {"thread_id": thread_id}}

        history = get_conversation_messages(self.db_session, conversation_id, limit=20)

        input_state = {
            "thread_id": thread_id,
            "influencer_phone": "",
            "agent_id": self.agent_config.agent_id,
            "last_user_message": user_message,
            "owner": "agent",
            "approval_required": False,
            "qualification_complete": False,
            "messages": [HumanMessage(content=user_message)],
            "current_node": "",
            "conversation_history": history,
            "influencer_id": influencer_id,
        }

        result = self.graph.invoke(input_state, config)

        # Persist influencer profile updates extracted during qualification
        if result.get("influencer_updates") and influencer_id:
            update_influencer_profile(
                self.db_session, influencer_id, **result["influencer_updates"]
            )
            self.db_session.commit()

        # Extract response from messages
        response = ""
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                    response = msg.content
                    break

        if not response:
            response = "Desculpe, não consegui processar sua mensagem. Pode repetir?"

        save_message(self.db_session, conversation_id, "assistant", response)
        self.db_session.commit()

        return {
            "response": response,
            "owner": result.get("owner", "agent"),
            "approval_required": result.get("approval_required", False),
        }

    def handle_approval(self, thread_id: str, decision: dict) -> dict:
        """Resume graph after approval interrupt."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(Command(resume=decision), config)

        response = ""
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                    response = msg.content
                    break

        return {
            "response": response or "Aprovação processada.",
            "owner": result.get("owner", "agent"),
            "approval_required": result.get("approval_required", False),
        }

    def close(self):
        """Cleanup resources."""
        self.db_session.close()
        self._checkpointer_ctx.__exit__(None, None, None)
