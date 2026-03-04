"""Streamlit web interface for the Negotiator Agent."""

import os

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets / env setup — must happen before any app import
# ---------------------------------------------------------------------------

def _setup_env():
    """Load secrets from Streamlit Cloud or .env file into os.environ."""
    from dotenv import load_dotenv
    load_dotenv()

    # Streamlit Cloud secrets override .env
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL", "DATABASE_URL", "CHECKPOINT_DB"):
        try:
            val = st.secrets[key]
            os.environ[key] = val
        except (KeyError, FileNotFoundError):
            pass

_setup_env()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agente Negociador — Gocase",
    page_icon="🤝",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Imports (after env setup)
# ---------------------------------------------------------------------------

from app.core.orchestrator import Orchestrator  # noqa: E402
from app.db.seed import seed as seed_db  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.db.models import Deal  # noqa: E402

# ---------------------------------------------------------------------------
# Auto-seed: populate benchmarks if the DB is empty
# ---------------------------------------------------------------------------

def _auto_seed():
    init_db()
    session = SessionLocal()
    try:
        if session.query(Deal).count() == 0:
            seed_db()
    finally:
        session.close()

_auto_seed()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "messages": [],
    "orchestrator": None,
    "thread_id": None,
    "conversation_id": None,
    "influencer_id": None,
    "approval_pending": False,
    "conversation_started": False,
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Helper: get or create orchestrator (one per browser session)
# ---------------------------------------------------------------------------

def _get_orchestrator() -> Orchestrator:
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = Orchestrator(agent_id="negotiator")
    return st.session_state.orchestrator

# ---------------------------------------------------------------------------
# Helper: start a new conversation
# ---------------------------------------------------------------------------

def _start_conversation(phone: str):
    orch = _get_orchestrator()

    # Close previous orchestrator if re-starting
    result = orch.start_or_resume_conversation(phone, new=True)

    st.session_state.thread_id = result["thread_id"]
    st.session_state.conversation_id = result["conversation"].id
    st.session_state.influencer_id = result["influencer"].id
    st.session_state.messages = []
    st.session_state.approval_pending = False
    st.session_state.conversation_started = True

    # Generate greeting
    greeting = orch.send_greeting(
        result["conversation"].id,
        influencer_name=result["influencer"].name,
    )

    st.session_state.messages.append({"role": "assistant", "content": greeting})

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Configuracoes")

    phone = st.text_input("Telefone do influenciador", value="+5585999999999")

    if st.button("Nova conversa", use_container_width=True):
        _start_conversation(phone)
        st.rerun()

    if st.button("Seed banco (popular dados)", use_container_width=True):
        count = seed_db()
        if count:
            st.success(f"{count} deals inseridos!")
        else:
            st.warning("Banco ja contem dados.")

    st.divider()
    st.caption("MVP — Agente Negociador Gocase")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Agente Negociador de Influenciadores")
st.caption("Negocie precos de conteudo com a Raimunda, agente virtual da Gocase.")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Approval panel (shown when approval is pending)
# ---------------------------------------------------------------------------

if st.session_state.approval_pending:
    st.warning("**Aprovacao necessaria** — A proposta requer aprovacao do operador.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Aprovar", use_container_width=True):
            orch = _get_orchestrator()
            approval = orch.handle_approval(
                st.session_state.thread_id,
                {"approved": True},
                conversation_id=st.session_state.conversation_id,
            )
            st.session_state.approval_pending = False
            st.session_state.messages.append(
                {"role": "assistant", "content": approval["response"]}
            )
            st.rerun()

    with col2:
        if st.button("Rejeitar", use_container_width=True):
            orch = _get_orchestrator()
            approval = orch.handle_approval(
                st.session_state.thread_id,
                {"approved": False},
                conversation_id=st.session_state.conversation_id,
            )
            st.session_state.approval_pending = False
            st.session_state.messages.append(
                {"role": "assistant", "content": approval["response"]}
            )
            st.rerun()

    with col3:
        counter_value = st.number_input(
            "Contraproposta (R$)", min_value=0.0, step=100.0, format="%.2f"
        )
        if st.button("Enviar contraproposta", use_container_width=True):
            orch = _get_orchestrator()
            approval = orch.handle_approval(
                st.session_state.thread_id,
                {"approved": False, "counter_offer_brl": counter_value},
                conversation_id=st.session_state.conversation_id,
            )
            st.session_state.approval_pending = False
            st.session_state.messages.append(
                {"role": "assistant", "content": approval["response"]}
            )
            st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if not st.session_state.conversation_started:
    st.info("Clique em **Nova conversa** na barra lateral para comecar.")
    st.stop()

if user_input := st.chat_input("Digite sua mensagem..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process through orchestrator
    orch = _get_orchestrator()

    with st.spinner("Pensando..."):
        response = orch.process_message(
            st.session_state.thread_id,
            st.session_state.conversation_id,
            user_input,
            influencer_id=st.session_state.influencer_id,
        )

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response["response"]}
    )
    with st.chat_message("assistant"):
        st.markdown(response["response"])

    # Handle special states
    if response["owner"] == "human":
        st.error("Conversa transferida para atendente humano.")

    if response["approval_required"]:
        st.session_state.approval_pending = True
        st.rerun()
