"""LangGraph graph for the negotiator agent."""

import json
import os

import openai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents.state import NegotiatorState
from app.tools import OPENAI_TOOL_SCHEMAS
from app.tools.guardrails import (
    HANDOFF_SUFFIX,
    SENSITIVE_RESPONSE,
    append_handoff_suffix,
    check_human_handoff,
    check_sensitive_data,
)
from app.tools.pricing import approval_required, calculate_price_range
from app.tools.retrieval import retrieve_benchmarks

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Você é um negociador profissional que trabalha para uma agência de marketing de influenciadores.

Seu objetivo é fechar o melhor negócio possível para a agência — ou seja, o MENOR preço que o influenciador aceitar.

Regras de negociação:
- NUNCA revele ao influenciador a faixa de preço interna (floor, target, ceiling). Isso é informação confidencial da agência.
- NUNCA mencione os termos "floor", "target", "ceiling", "faixa de preço" ou valores mínimos/máximos que você está disposto a pagar.
- Sempre COMECE propondo um valor próximo ao floor (o menor valor aceitável).
- Use benchmarks de mercado como justificativa ("deals similares no mercado fecham por volta de R$X").
- Se o influenciador pedir mais, suba gradualmente, nunca mais que 10-15% por rodada.
- Só ultrapasse o ceiling se receber aprovação do operador.
- Seja cordial, profissional e persuasivo.
- Se o influenciador pedir para falar com um humano, respeite imediatamente.
- Nunca colete dados sensíveis (CPF, cartão de crédito, senhas).
- Sempre responda em português brasileiro.

Estratégia:
- Justifique suas propostas com dados de mercado, não com limites internos.
- Se o influenciador propor um valor alto, contra-argumente com benchmarks.
- Mostre flexibilidade mas sempre proteja o orçamento da agência.
- Destaque o valor da parceria de longo prazo como argumento para preços menores.

{context}"""

QUALIFICATION_FIELDS = ["platform", "deliverable_type", "avg_views", "qty", "deadline"]

EXTRACT_INFO_TOOL = {
    "type": "function",
    "name": "extract_info",
    "description": (
        "Extrai informações estruturadas da mensagem do influenciador. "
        "Chame esta função SEMPRE que a mensagem contiver qualquer uma dessas informações, "
        "mesmo que parcialmente."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "enum": ["instagram", "tiktok", "youtube"],
                "description": "Plataforma mencionada pelo influenciador",
            },
            "deliverable_type": {
                "type": "string",
                "enum": ["reel", "post", "story", "video"],
                "description": "Tipo de entregável (reel, post, story, video)",
            },
            "avg_views": {
                "type": "integer",
                "description": "Média de visualizações por conteúdo. Converter '100k' para 100000, '50k' para 50000, etc.",
            },
            "qty": {
                "type": "integer",
                "description": "Quantidade de peças/conteúdos",
            },
            "deadline": {
                "type": "string",
                "description": "Prazo de entrega mencionado",
            },
            "niche": {
                "type": "string",
                "description": "Nicho do influenciador se mencionado",
            },
            "proposed_price_brl": {
                "type": "number",
                "description": "Preço proposto pelo influenciador em reais, se mencionado. Converter '100k' para 100000.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


def _build_context(state: NegotiatorState) -> str:
    """Build context string for the system prompt.

    Uses internal labels so the model knows the values but doesn't leak them.
    """
    parts = []
    if state.get("suggested_range"):
        r = state["suggested_range"]
        parts.append(
            "[DADOS INTERNOS — NUNCA revelar estes valores ao influenciador]\n"
            f"  Valor mínimo aceitável (comece aqui): R${r['floor']:.2f}\n"
            f"  Valor ideal para a agência: R${r['target']:.2f}\n"
            f"  Valor máximo (precisa de aprovação acima): R${r['ceiling']:.2f}"
        )
    if state.get("benchmarks") and state["benchmarks"].get("count", 0) > 0:
        b = state["benchmarks"]
        parts.append(
            "[DADOS DE MERCADO — pode usar para justificar propostas]\n"
            f"  {b['count']} deals similares encontrados\n"
            f"  CPM médio de mercado: R${b['avg_cpm']:.2f}\n"
            f"  Preço mediano de mercado: R${b['median_price']:.2f}"
        )
    if state.get("current_offer_brl"):
        parts.append(
            f"[PROPOSTA DO INFLUENCIADOR]: R${state['current_offer_brl']:.2f}"
        )
    known = {f: state.get(f) for f in QUALIFICATION_FIELDS if state.get(f)}
    if known:
        parts.append(f"Dados coletados do influenciador: {known}")
    return "\n".join(parts)


def _extract_text(output_items) -> str:
    """Extract text content from OpenAI response output items."""
    texts = []
    for item in output_items:
        if hasattr(item, "type") and item.type == "message":
            for block in item.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
        elif hasattr(item, "text"):
            texts.append(item.text)
    return "\n".join(texts) if texts else ""


def _dispatch_tool(name: str, arguments: dict, session=None) -> dict:
    """Dispatch a tool call to the appropriate function."""
    if name == "retrieve_benchmarks":
        return retrieve_benchmarks(
            platform=arguments["platform"],
            deliverable_type=arguments["deliverable_type"],
            avg_views=arguments["avg_views"],
            niche=arguments.get("niche"),
            k=arguments.get("k", 5),
            session=session,
        )
    elif name == "calculate_price_range":
        return calculate_price_range(
            avg_views=arguments["avg_views"],
            qty=arguments["qty"],
            target_cpm_brl=arguments["target_cpm_brl"],
            benchmarks=arguments.get("benchmarks"),
        )
    elif name == "check_approval_required":
        return {
            "approval_required": approval_required(
                proposed_brl=arguments["proposed_brl"],
                price_range=arguments["price_range"],
                benchmarks=arguments.get("benchmarks"),
            )
        }
    return {"error": f"Unknown tool: {name}"}


def _run_openai_with_tools(
    conversation: list,
    tools: list,
    system_prompt: str,
    model: str = MODEL,
    session=None,
) -> tuple[str, list]:
    """Run OpenAI Responses API with tool loop."""
    client = openai.OpenAI()

    for _attempt in range(10):  # max tool loops
        response = client.responses.create(
            model=model,
            input=conversation,
            instructions=system_prompt,
            tools=tools,
        )

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if not function_calls:
            text = _extract_text(response.output)
            return text, conversation

        for fc in function_calls:
            conversation.append(fc.to_dict())
            result = _dispatch_tool(fc.name, json.loads(fc.arguments), session)
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": json.dumps(result),
                }
            )

    text = _extract_text(response.output)
    return text, conversation


# ── LangGraph Nodes ──────────────────────────────────────────────


def _extract_fields_from_message(user_message: str, state: NegotiatorState) -> dict:
    """Use OpenAI to extract structured fields from user message."""
    client = openai.OpenAI()

    known = {f: state.get(f) for f in QUALIFICATION_FIELDS if state.get(f)}
    known_str = json.dumps(known, ensure_ascii=False) if known else "nenhum"

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": (
                    f"Mensagem do influenciador: '{user_message}'\n\n"
                    f"Dados já coletados: {known_str}\n\n"
                    "Extraia TODAS as informações presentes na mensagem usando a tool extract_info. "
                    "Converta valores como '100k' para 100000. "
                    "Se o influenciador menciona 'reels', o deliverable_type é 'reel' e a platform é 'instagram'. "
                    "Se menciona 'stories', o deliverable_type é 'story'. "
                    "SEMPRE chame a tool se houver qualquer informação extraível."
                ),
            }
        ],
        instructions="Você é um extrator de dados. Extraia informações estruturadas usando a tool disponível.",
        tools=[EXTRACT_INFO_TOOL],
    )

    extracted = {}
    for item in response.output:
        if item.type == "function_call" and item.name == "extract_info":
            extracted = json.loads(item.arguments)
            break

    return extracted


def qualify(state: NegotiatorState) -> dict:
    """Extract info from user message, then ask for missing fields."""
    # Step 1: Extract data from user message
    extracted = _extract_fields_from_message(
        state["last_user_message"], state
    )

    # Build state updates from extracted data
    updates = {}
    field_mapping = {
        "platform": "platform",
        "deliverable_type": "deliverable_type",
        "avg_views": "avg_views",
        "qty": "qty",
        "deadline": "deadline",
        "niche": "niche",
    }
    for ext_key, state_key in field_mapping.items():
        if extracted.get(ext_key) and not state.get(state_key):
            updates[state_key] = extracted[ext_key]

    if extracted.get("proposed_price_brl"):
        updates["current_offer_brl"] = extracted["proposed_price_brl"]

    # Step 2: Check what's still missing after extraction
    merged = {**{f: state.get(f) for f in QUALIFICATION_FIELDS}, **updates}
    missing = [f for f in QUALIFICATION_FIELDS if not merged.get(f)]

    if not missing:
        updates["qualification_complete"] = True
        updates["current_node"] = "qualify"
        return updates

    # Step 3: Generate natural question for missing fields
    context = _build_context(state)
    known_now = {f: merged[f] for f in QUALIFICATION_FIELDS if merged.get(f)}
    missing_str = ", ".join(missing)

    system = SYSTEM_PROMPT.format(context=context)
    conversation = [
        {
            "role": "user",
            "content": (
                f"O influenciador disse: '{state['last_user_message']}'\n\n"
                f"Dados já confirmados: {json.dumps(known_now, ensure_ascii=False)}\n"
                f"Campos ainda faltando: {missing_str}.\n\n"
                "Primeiro confirme os dados que você já coletou de forma breve e natural. "
                "Depois pergunte APENAS o que falta. Seja conciso e cordial."
            ),
        }
    ]

    text, _ = _run_openai_with_tools(conversation, [], system)
    text = append_handoff_suffix(text)

    updates["messages"] = [AIMessage(content=text)]
    updates["current_node"] = "qualify"
    updates["qualification_complete"] = False

    return updates


def retrieve_benchmarks_node(state: NegotiatorState) -> dict:
    """Deterministic node: query benchmarks from DB."""
    benchmarks = retrieve_benchmarks(
        platform=state.get("platform", "instagram"),
        deliverable_type=state.get("deliverable_type", "reel"),
        avg_views=state.get("avg_views", 50000),
        niche=state.get("niche"),
    )
    return {"benchmarks": benchmarks, "current_node": "retrieve_benchmarks"}


def price_node(state: NegotiatorState) -> dict:
    """Deterministic node: calculate price range."""
    target_cpm = state.get("target_cpm_brl", 40.0)  # default CPM
    price_range = calculate_price_range(
        avg_views=state.get("avg_views", 50000),
        qty=state.get("qty", 1),
        target_cpm_brl=target_cpm,
        benchmarks=state.get("benchmarks"),
    )
    return {"suggested_range": price_range, "current_node": "price"}


def negotiate(state: NegotiatorState) -> dict:
    """Core LLM negotiation node with tool calling."""
    context = _build_context(state)
    system = SYSTEM_PROMPT.format(context=context)

    conversation = [
        {
            "role": "user",
            "content": state["last_user_message"],
        }
    ]

    text, _ = _run_openai_with_tools(
        conversation, OPENAI_TOOL_SCHEMAS, system
    )
    text = append_handoff_suffix(text)

    needs_approval = False
    if state.get("current_offer_brl") and state.get("suggested_range"):
        needs_approval = approval_required(
            state["current_offer_brl"],
            state["suggested_range"],
            state.get("benchmarks"),
        )

    return {
        "messages": [AIMessage(content=text)],
        "approval_required": needs_approval,
        "current_node": "negotiate",
    }


def approval_node(state: NegotiatorState) -> dict:
    """Interrupt for human approval."""
    decision = interrupt(
        {
            "type": "approval_required",
            "current_offer_brl": state.get("current_offer_brl"),
            "suggested_range": state.get("suggested_range"),
            "message": "Aprovação necessária. Aceitar esta proposta? (sim/não/contraproposta)",
        }
    )

    if isinstance(decision, dict):
        approved = decision.get("approved", False)
        counter = decision.get("counter_offer_brl")
    else:
        approved = str(decision).lower() in ("sim", "yes", "s", "true")
        counter = None

    if counter:
        return {
            "current_offer_brl": counter,
            "approval_required": False,
            "current_node": "approval",
            "messages": [
                AIMessage(content=f"Operador ajustou a proposta para R${counter:.2f}.")
            ],
        }
    elif approved:
        return {
            "approval_required": False,
            "current_node": "approval",
            "messages": [AIMessage(content="Proposta aprovada pelo operador.")],
        }
    else:
        return {
            "approval_required": False,
            "current_node": "approval",
            "messages": [
                AIMessage(content="Proposta recusada pelo operador. Renegociando...")
            ],
        }


def close_node(state: NegotiatorState) -> dict:
    """Generate deal summary and close conversation."""
    summary = {
        "thread_id": state["thread_id"],
        "platform": state.get("platform"),
        "deliverable_type": state.get("deliverable_type"),
        "qty": state.get("qty"),
        "avg_views": state.get("avg_views"),
        "final_offer_brl": state.get("current_offer_brl"),
        "price_range": state.get("suggested_range"),
        "status": "closed",
    }

    text = (
        f"Negociação finalizada!\n\n"
        f"Resumo: {json.dumps(summary, ensure_ascii=False, indent=2)}"
    )

    return {
        "messages": [AIMessage(content=text)],
        "current_node": "close",
        "owner": "agent",
    }


# ── Conditional edges ────────────────────────────────────────────


def after_qualify(state: NegotiatorState) -> str:
    if state.get("owner") == "human":
        return "close"
    if not state.get("qualification_complete"):
        return END  # wait for user reply
    if state.get("suggested_range"):
        return "negotiate"  # re-entry optimization
    return "retrieve_benchmarks"


def after_negotiate(state: NegotiatorState) -> str:
    if state.get("owner") == "human":
        return "close"
    if state.get("approval_required"):
        return "approval"
    return END  # wait for user reply


def after_approval(state: NegotiatorState) -> str:
    return "negotiate"


# ── Build graph ──────────────────────────────────────────────────


def build_graph(checkpointer=None):
    """Build and compile the negotiator LangGraph."""
    graph = StateGraph(NegotiatorState)

    graph.add_node("qualify", qualify)
    graph.add_node("retrieve_benchmarks", retrieve_benchmarks_node)
    graph.add_node("price", price_node)
    graph.add_node("negotiate", negotiate)
    graph.add_node("approval", approval_node)
    graph.add_node("close", close_node)

    graph.add_edge(START, "qualify")
    graph.add_conditional_edges("qualify", after_qualify)
    graph.add_edge("retrieve_benchmarks", "price")
    graph.add_edge("price", "negotiate")
    graph.add_conditional_edges("negotiate", after_negotiate)
    graph.add_conditional_edges("approval", after_approval)
    graph.add_edge("close", END)

    return graph.compile(checkpointer=checkpointer)
