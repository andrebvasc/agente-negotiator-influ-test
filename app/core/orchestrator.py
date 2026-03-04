"""Orquestrador: ponte entre CLI e LangGraph."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.negotiator import (
    PERSONAL_INFO_FIELDS,
    build_graph,
    extract_personal_info,
    generate_greeting,
    generate_post_deal_response,
)
from app.core.registry import registry
from app.core.store import (
    create_conversation,
    get_active_conversation,
    get_conversation_messages,
    get_or_create_agent,
    get_or_create_influencer,
    save_deal,
    save_message,
    update_conversation_owner,
    update_conversation_status,
    update_influencer_profile,
)
from app.db.session import SessionLocal, init_db
from app.tools.guardrails import check_human_handoff, check_sensitive_data, SENSITIVE_RESPONSE

load_dotenv()

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite")


class Orchestrator:
    """Gerencia o ciclo de vida das conversas entre CLI e LangGraph."""

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
        """Inicia uma nova conversa ou retoma uma existente."""
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

    def send_greeting(
        self,
        conversation_id: int,
        user_message: str | None = None,
        influencer_name: str | None = None,
    ) -> str:
        """Gera e persiste a saudação inicial para uma nova conversa.

        Se o influenciador já tem nome cadastrado, a saudação pula a
        auto-apresentação. Se *user_message* é fornecido, a saudação é
        adaptada para respondê-lo.
        """
        greeting = generate_greeting(
            user_message=user_message,
            influencer_name=influencer_name,
        )
        save_message(self.db_session, conversation_id, "assistant", greeting)
        self.db_session.commit()
        return greeting

    def _process_post_deal(
        self, conversation_id: int, user_message: str, influencer_id: int | None
    ) -> dict:
        """Trata mensagens após fechamento de deal — coleta dados pessoais."""
        save_message(self.db_session, conversation_id, "user", user_message)
        self.db_session.commit()

        # Carrega campos pessoais já conhecidos do influenciador
        from app.db.models import Influencer

        known = {}
        inf = None
        if influencer_id:
            inf = self.db_session.query(Influencer).get(influencer_id)
            if inf:
                for f in PERSONAL_INFO_FIELDS:
                    val = getattr(inf, f, None)
                    if val:
                        known[f] = val

        # Extrai novos dados pessoais da mensagem
        extracted = extract_personal_info(user_message, known)

        # Faz merge dos dados extraídos no registro do influenciador
        if extracted and influencer_id:
            update_influencer_profile(self.db_session, influencer_id, **extracted)
            self.db_session.commit()
            known.update({k: v for k, v in extracted.items() if v})

        # Checa o que ainda falta
        missing = [f for f in PERSONAL_INFO_FIELDS if not known.get(f)]

        influencer_name = inf.name if inf else None
        response = generate_post_deal_response(
            user_message, known, missing, influencer_name
        )

        # Se todos os dados foram coletados, marca conversa como completa
        if not missing:
            from app.core.store import update_conversation_status

            update_conversation_status(self.db_session, conversation_id, "completed")
            self.db_session.commit()

        save_message(self.db_session, conversation_id, "assistant", response)
        self.db_session.commit()

        return {
            "response": response,
            "owner": "agent",
            "approval_required": False,
        }

    def process_message(
        self,
        thread_id: str,
        conversation_id: int,
        user_message: str,
        influencer_id: int | None = None,
    ) -> dict:
        """Processa uma mensagem do usuário pelo grafo."""
        # Fase pós-deal: coleta dados pessoais ao invés de rodar o grafo
        from app.db.models import Conversation

        conv = self.db_session.query(Conversation).get(conversation_id)
        if conv and conv.status == "closed_deal":
            return self._process_post_deal(
                conversation_id, user_message, influencer_id
            )

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

        # Carrega perfil existente do influenciador para pré-popular campos conhecidos
        influencer_profile = {}
        if influencer_id:
            from app.db.models import Influencer
            inf = self.db_session.query(Influencer).get(influencer_id)
            if inf:
                for field in ("name", "platform", "niche", "avg_views"):
                    val = getattr(inf, field, None)
                    if val:
                        influencer_profile[field] = val

        # Resolve telefone do influenciador a partir do registro
        _inf_phone = ""
        if influencer_id:
            from app.db.models import Influencer as _Inf
            _inf_obj = self.db_session.query(_Inf).get(influencer_id)
            if _inf_obj:
                _inf_phone = _inf_obj.phone or ""

        input_state = {
            "thread_id": thread_id,
            "influencer_phone": _inf_phone,
            "agent_id": self.agent_config.agent_id,
            "last_user_message": user_message,
            "owner": "agent",
            "approval_required": False,
            "qualification_complete": False,
            "deal_accepted": False,
            "messages": [HumanMessage(content=user_message)],
            "current_node": "",
            "conversation_history": history,
            "influencer_id": influencer_id,
            **influencer_profile,
        }

        result = self.graph.invoke(input_state, config)

        # Persiste atualizações do perfil extraídas durante qualificação
        if result.get("influencer_updates") and influencer_id:
            update_influencer_profile(
                self.db_session, influencer_id, **result["influencer_updates"]
            )
            self.db_session.commit()

        # Persiste deal(s) se o influenciador aceitou
        if result.get("deal_to_save"):
            deals = result["deal_to_save"]
            if isinstance(deals, dict):
                deals = [deals]
            for deal_data in deals:
                save_deal(self.db_session, deal_data)
            update_conversation_status(self.db_session, conversation_id, "closed_deal")
            self.db_session.commit()

        # Extrai resposta das mensagens
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

    def handle_approval(
        self, thread_id: str, decision: dict, conversation_id: int | None = None
    ) -> dict:
        """Retoma o grafo após interrupção de aprovação."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(Command(resume=decision), config)

        # Persiste deal(s) se aprovação levou ao fechamento
        if result.get("deal_to_save") and conversation_id:
            deals = result["deal_to_save"]
            if isinstance(deals, dict):
                deals = [deals]
            for deal_data in deals:
                save_deal(self.db_session, deal_data)
            update_conversation_status(self.db_session, conversation_id, "closed_deal")
            self.db_session.commit()

        response = ""
        if result.get("messages"):
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                    response = msg.content
                    break

        if response and conversation_id:
            save_message(self.db_session, conversation_id, "assistant", response)
            self.db_session.commit()

        return {
            "response": response or "Aprovação processada.",
            "owner": result.get("owner", "agent"),
            "approval_required": result.get("approval_required", False),
        }

    def close(self):
        """Libera recursos."""
        self.db_session.close()
        self._checkpointer_ctx.__exit__(None, None, None)
