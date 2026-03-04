"""LangGraph graph for the negotiator agent."""

import json
import os
import re

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

SYSTEM_PROMPT = """Você é a Raimunda, negociadora profissional que trabalha para a Gocase.

Seu objetivo é fechar o melhor negócio possível para a agência — ou seja, o MENOR preço que o influenciador aceitar.

Regras de negociação:
- NUNCA revele ao influenciador a faixa de preço interna (floor, target, ceiling). Isso é informação confidencial da agência.
- NUNCA mencione os termos "floor", "target", "ceiling", "faixa de preço" ou valores mínimos/máximos que você está disposto a pagar.
- NUNCA proponha um valor em reais (R$) por iniciativa própria. Você NÃO sugere preços. Quem define preço é o influenciador.
- A ÚNICA exceção é quando há uma [CONTRAPROPOSTA DO OPERADOR] nos dados internos — nesse caso, apresente exatamente aquele valor.
- Use benchmarks de mercado como ARGUMENTO para convencer o influenciador a baixar o preço, mas sem propor um número específico. Ex: "deals similares no mercado fecham por valores bem abaixo disso" ou "o mercado pratica valores mais acessíveis para esse tipo de parceria".
- Só ultrapasse o ceiling se receber aprovação do operador.
- Seja cordial, profissional e persuasivo.
- Se o influenciador pedir para falar com um humano, respeite imediatamente.
- Nunca colete dados sensíveis (cartão de crédito, senhas).
- Sempre responda em português brasileiro.

Estratégia:
- SEMPRE espere o influenciador dizer o preço dele primeiro. Pergunte: "Qual seria o seu valor para essa parceria?" ou similar.
- Só depois que o influenciador informar o preço, reaja com base nos seus dados internos.
- Se o valor do influenciador estiver dentro da faixa (floor-ceiling), aceite.
- Se o valor estiver acima do ceiling, NÃO faça contraproposta com valor. Em vez disso:
  - Diga que está acima do que o mercado pratica para parcerias similares.
  - Peça ao influenciador para reconsiderar e informar o mínimo que aceitaria.
  - Destaque a parceria de longo prazo como argumento.
  - NUNCA diga "que tal R$X?" ou "posso oferecer R$X" por conta própria.
- Mostre flexibilidade mas sempre proteja o orçamento da agência.

Aceitação de deal:
- Quando o influenciador ACEITAR explicitamente uma proposta de preço (ex: "fechado", "aceito", "pode ser", "ok", "sim", "vamos nesse valor", "tá bom", "beleza"), chame IMEDIATAMENTE a tool `confirm_deal` com accepted=true e o valor acordado em agreed_price_brl.
- O valor acordado é a ÚLTIMA proposta que estava em discussão (seja a que você propôs ou a que o influenciador propôs).
- Respostas curtas como "sim", "ok", "fechado" em resposta a uma proposta de preço SÃO aceitações — chame confirm_deal.
- Só NÃO chame confirm_deal se houver dúvida genuína ou contra-proposta explícita.

{context}"""

GREETING_NEW = (
    "Olá, sou a Raimunda, tudo bem com você? 😊\n\n"
    "Tô aqui para lhe auxiliar na parceria com a Gocase!\n\n"
    "Para começarmos, pode me dizer seu nome e me enviar seus valores "
    "e formatos disponíveis? Assim já alinhamos a melhor proposta pra você.\n\n"
    "Se preferir, também posso te passar mais detalhes da marca, "
    "objetivos da campanha e expectativas de entrega.\n\n"
    "Fico no aguardo! ✨"
)

GREETING_SYSTEM = (
    "Você é a Raimunda, negociadora profissional da Gocase. "
    "Responda sempre em português brasileiro. "
    "Mantenha o tom amigável e profissional. Use emoji com moderação (1-2 no máximo)."
)

QUALIFICATION_FIELDS = ["name", "platform", "deliverable_type", "avg_views", "qty", "deadline"]

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
            "name": {
                "type": "string",
                "description": "Nome do influenciador, se mencionado na mensagem.",
            },
            "platform": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["instagram", "tiktok", "youtube"],
                },
                "description": "Plataformas mencionadas pelo influenciador (pode ser mais de uma)",
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


CONFIRM_DEAL_TOOL = {
    "type": "function",
    "name": "confirm_deal",
    "description": (
        "Chame esta função quando o influenciador ACEITAR explicitamente o preço proposto. "
        "Só chame quando houver aceitação clara (ex: 'fechado', 'aceito', 'ok')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "accepted": {
                "type": "boolean",
                "description": "True se o influenciador aceitou o deal",
            },
            "agreed_price_brl": {
                "type": "number",
                "description": "Valor final acordado em reais (BRL)",
            },
        },
        "required": ["accepted", "agreed_price_brl"],
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
    if state.get("last_agent_offer_brl"):
        parts.append(
            f"[SUA ÚLTIMA PROPOSTA AO INFLUENCIADOR]: R${state['last_agent_offer_brl']:.2f}"
        )
    if state.get("current_offer_brl"):
        parts.append(
            f"[PROPOSTA DO INFLUENCIADOR]: R${state['current_offer_brl']:.2f}"
        )
    else:
        parts.append(
            "[O INFLUENCIADOR AINDA NÃO INFORMOU SEU PREÇO] — Pergunte qual o valor dele antes de qualquer proposta."
        )
    if state.get("operator_counter_offer_brl"):
        parts.append(
            f"[CONTRAPROPOSTA DO OPERADOR]: R${state['operator_counter_offer_brl']:.2f}\n"
            "AÇÃO OBRIGATÓRIA: Apresente EXATAMENTE este valor como sua oferta ao influenciador. "
            "Diga algo como 'Consigo te oferecer R$X.XXX por essa parceria, o que acha?'. "
            "NÃO pergunte o mínimo do influenciador — OFEREÇA este valor diretamente."
        )
    known = {f: state.get(f) for f in QUALIFICATION_FIELDS if state.get(f)}
    if state.get("name"):
        known["name"] = state["name"]
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


def generate_greeting(
    user_message: str | None = None,
    influencer_name: str | None = None,
) -> str:
    """Return the opening greeting for a new conversation.

    Three scenarios:
    1. Known influencer (has name) → returning-contact greeting, no self-intro.
    2. New influencer + first message → adapted greeting with self-intro + answer.
    3. New influencer, no message → default ``GREETING_NEW``.
    """
    is_known = bool(influencer_name)
    has_message = bool(user_message and user_message.strip())

    # Scenario 3: brand-new contact, no message yet
    if not is_known and not has_message:
        return GREETING_NEW

    # Scenarios 1 & 2: use LLM to craft the right tone
    parts = []
    if is_known:
        parts.append(
            f"O influenciador já te conhece de conversas anteriores. O nome dele(a) é {influencer_name}.\n"
            "NÃO se apresente novamente — vá direto a uma saudação calorosa usando o nome dele(a), "
            "como se fosse um reencontro. Exemplo: 'Oi {name}, que bom falar com você de novo!'\n"
        )
    else:
        parts.append(
            "Este é um primeiro contato. Apresente-se brevemente como Raimunda da Gocase.\n"
        )

    if has_message:
        parts.append(
            f"O influenciador já enviou uma mensagem: '{user_message}'\n"
            "Responda ao que ele disse/perguntou de forma natural ANTES de pedir valores e formatos.\n"
        )

    if not is_known:
        parts.append(
            "Peça o nome do influenciador caso ele ainda não tenha se apresentado.\n"
        )

    parts.append(
        "Depois peça os valores e formatos disponíveis para alinhar a proposta.\n"
        "Seja concisa (máximo 4-5 linhas)."
    )

    prompt = "\n".join(parts)

    client = openai.OpenAI()
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        instructions=GREETING_SYSTEM,
    )
    text = _extract_text(response.output)
    return text or GREETING_NEW


# ── Post-deal personal info collection ───────────────────────────

PERSONAL_INFO_FIELDS = ["email", "cpf", "address", "phone_model"]

EXTRACT_PERSONAL_TOOL = {
    "type": "function",
    "name": "extract_personal_info",
    "description": "Extrai dados pessoais da mensagem do influenciador (email, CPF, endereço, modelo de celular).",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "Email do influenciador",
            },
            "cpf": {
                "type": "string",
                "description": "CPF do influenciador (formato XXX.XXX.XXX-XX ou só números)",
            },
            "address": {
                "type": "string",
                "description": "Endereço completo do influenciador",
            },
            "phone_model": {
                "type": "string",
                "description": "Modelo do celular do influenciador (ex: iPhone 15, Samsung S24)",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


def extract_personal_info(user_message: str, known: dict) -> dict:
    """Extract personal info fields from user message using LLM."""
    client = openai.OpenAI()
    known_str = json.dumps(known, ensure_ascii=False) if known else "nenhum"

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": (
                    f"Mensagem do influenciador: '{user_message}'\n\n"
                    f"Dados já coletados: {known_str}\n\n"
                    "Extraia email, CPF, endereço e modelo de celular da mensagem. "
                    "Se não houver nenhum dado pessoal, NÃO chame a tool."
                ),
            }
        ],
        instructions="Você é um extrator de dados. Extraia informações pessoais usando a tool disponível.",
        tools=[EXTRACT_PERSONAL_TOOL],
    )

    extracted = {}
    for item in response.output:
        if item.type == "function_call" and item.name == "extract_personal_info":
            extracted = json.loads(item.arguments)
            break
    return extracted


def generate_post_deal_response(
    user_message: str, known: dict, missing: list[str], influencer_name: str | None
) -> str:
    """Generate a natural response during post-deal info collection."""
    name = influencer_name or "você"

    if not missing:
        return (
            f"Perfeito, {name}! Recebi todos os dados. "
            "Muito obrigada! Em breve entraremos em contato com os próximos passos. "
            "Tenha um ótimo dia! 😊"
        )

    field_labels = {
        "email": "email",
        "cpf": "CPF",
        "address": "endereço completo",
        "phone_model": "modelo do celular",
    }
    missing_labels = [field_labels.get(f, f) for f in missing]

    if len(missing_labels) == 1:
        missing_str = missing_labels[0]
    else:
        missing_str = ", ".join(missing_labels[:-1]) + " e " + missing_labels[-1]

    client = openai.OpenAI()
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": (
                    f"O influenciador ({name}) enviou: '{user_message}'\n"
                    f"Dados já recebidos: {json.dumps(known, ensure_ascii=False)}\n"
                    f"Ainda falta receber: {missing_str}\n\n"
                    "Agradeça os dados recebidos de forma breve e peça o que falta. "
                    "Seja cordial e concisa."
                ),
            }
        ],
        instructions=GREETING_SYSTEM,
    )
    text = _extract_text(response.output)
    return text or f"Obrigada! Só falta o {missing_str}. Pode me enviar?"


def _dispatch_tool(name: str, arguments: dict, session=None, deal_result: dict | None = None) -> dict:
    """Dispatch a tool call to the appropriate function."""
    if name == "confirm_deal":
        if deal_result is not None:
            deal_result["accepted"] = arguments.get("accepted", False)
            deal_result["agreed_price_brl"] = arguments.get("agreed_price_brl")
        return {"status": "deal_confirmed", "accepted": arguments.get("accepted", False)}
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
    deal_result: dict | None = None,
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
            result = _dispatch_tool(
                fc.name, json.loads(fc.arguments), session, deal_result
            )
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
                    "IMPORTANTE sobre o campo 'name': só extraia um nome se for claramente um nome próprio de pessoa "
                    "(ex: 'Sou a Maria', 'Meu nome é João'). "
                    "NÃO extraia cumprimentos ('oi', 'olá', 'e aí'), perguntas ou palavras genéricas como nome. "
                    "Na dúvida, NÃO preencha o campo name. "
                    "Se não houver nenhuma informação extraível, NÃO chame a tool."
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
    influencer_updates = {}
    field_mapping = {
        "name": "name",
        "deliverable_type": "deliverable_type",
        "avg_views": "avg_views",
        "qty": "qty",
        "deadline": "deadline",
        "niche": "niche",
    }
    for ext_key, state_key in field_mapping.items():
        if extracted.get(ext_key) and not state.get(state_key):
            updates[state_key] = extracted[ext_key]

    # Platform: merge as comma-separated set
    if extracted.get("platform"):
        new_platforms = extracted["platform"]
        if isinstance(new_platforms, str):
            new_platforms = [new_platforms]
        existing = set(state.get("platform", "").split(",")) if state.get("platform") else set()
        existing.discard("")
        merged = existing | set(new_platforms)
        updates["platform"] = ",".join(sorted(merged))

    # Collect fields to persist to the influencers table
    persist_fields = ("name", "niche", "avg_views")
    for field in persist_fields:
        if extracted.get(field):
            influencer_updates[field] = extracted[field]
    # Platform persisted as merged comma-separated
    if updates.get("platform"):
        influencer_updates["platform"] = updates["platform"]
    if influencer_updates:
        updates["influencer_updates"] = influencer_updates

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

    # Build conversation with history for context continuity
    conversation = []
    for msg in (state.get("conversation_history") or []):
        conversation.append({"role": msg["role"], "content": msg["content"]})
    # Add instruction as the last user message
    conversation.append(
        {
            "role": "user",
            "content": (
                f"O influenciador disse: '{state['last_user_message']}'\n\n"
                f"Dados já confirmados: {json.dumps(known_now, ensure_ascii=False)}\n"
                f"Campos ainda faltando: {missing_str}.\n\n"
                "Se o influenciador fez uma pergunta (como perguntar seu nome ou dados), "
                "responda usando os dados confirmados acima ANTES de pedir o que falta. "
                "Confirme os dados coletados de forma breve e natural, "
                "depois pergunte APENAS o que falta. Seja conciso e cordial."
            ),
        }
    )

    text, _ = _run_openai_with_tools(conversation, [], system)
    text = append_handoff_suffix(text)

    updates["messages"] = [AIMessage(content=text)]
    updates["current_node"] = "qualify"
    updates["qualification_complete"] = False

    return updates


def retrieve_benchmarks_node(state: NegotiatorState) -> dict:
    """Deterministic node: query benchmarks from DB."""
    platform_raw = state.get("platform", "instagram")
    first_platform = platform_raw.split(",")[0] if platform_raw else "instagram"
    benchmarks = retrieve_benchmarks(
        platform=first_platform,
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


def _extract_price_from_text(text: str) -> float | None:
    """Extract the last R$ price mentioned in agent's response."""
    matches = re.findall(r"R\$\s?([\d.,]+)", text)
    if not matches:
        return None
    raw = matches[-1].replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def negotiate(state: NegotiatorState) -> dict:
    """Core LLM negotiation node with tool calling."""
    context = _build_context(state)
    system = SYSTEM_PROMPT.format(context=context)

    # Build conversation with history for context continuity
    conversation = []
    for msg in (state.get("conversation_history") or []):
        conversation.append({"role": msg["role"], "content": msg["content"]})
    # Append current message if not already the last in history
    if not conversation or conversation[-1]["content"] != state["last_user_message"]:
        conversation.append({"role": "user", "content": state["last_user_message"]})

    negotiate_tools = OPENAI_TOOL_SCHEMAS + [CONFIRM_DEAL_TOOL]
    deal_result = {}

    text, _ = _run_openai_with_tools(
        conversation, negotiate_tools, system, deal_result=deal_result
    )
    text = append_handoff_suffix(text)

    needs_approval = False
    if state.get("current_offer_brl") and state.get("suggested_range"):
        needs_approval = approval_required(
            state["current_offer_brl"],
            state["suggested_range"],
            state.get("benchmarks"),
        )

    result = {
        "messages": [AIMessage(content=text)],
        "approval_required": needs_approval,
        "current_node": "negotiate",
        "operator_counter_offer_brl": None,  # clear after use
    }

    # Track the last price the agent proposed so we know the value in play
    agent_offer = _extract_price_from_text(text)
    if agent_offer:
        result["last_agent_offer_brl"] = agent_offer

    # If confirm_deal was called, flag the deal as accepted
    if deal_result.get("accepted"):
        result["deal_accepted"] = True
        result["agreed_price_brl"] = deal_result.get("agreed_price_brl")

    return result


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
        # Update ceiling to operator's counter-offer so the agent can propose it
        updated_range = dict(state.get("suggested_range") or {})
        updated_range["ceiling"] = counter
        return {
            "suggested_range": updated_range,
            "operator_counter_offer_brl": counter,
            "approval_required": False,
            "current_node": "approval",
            "messages": [
                AIMessage(content=f"Operador definiu contraproposta de R${counter:.2f}.")
            ],
        }
    elif approved:
        return {
            "approval_required": False,
            "deal_accepted": True,
            "agreed_price_brl": state.get("current_offer_brl"),
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


DEAL_CLOSING_MESSAGE = (
    "Ótimo, {name}! Fico feliz que chegamos em um acordo!\n\n"
    "Para o envio dos produtos e da proposta, poderia me informar:\n"
    "- Seu email\n"
    "- CPF\n"
    "- Endereço completo\n"
    "- Modelo do seu celular\n\n"
    "Em breve entraremos em contato novamente para passar os próximos passos! "
    "Tenha uma excelente semana e vamos juntos nessa nova parceria! ✨"
)


def save_deal_node(state: NegotiatorState) -> dict:
    """Build deal summary for the orchestrator to persist."""
    platform_raw = state.get("platform", "instagram")
    first_platform = platform_raw.split(",")[0] if platform_raw else "instagram"
    avg_views = state.get("avg_views", 0)
    agreed_price = state.get("agreed_price_brl", 0.0)
    qty = state.get("qty", 1)

    cpm = (agreed_price / (avg_views / 1000 * qty)) if avg_views and qty else 0.0

    deal = {
        "influencer_name": state.get("name", ""),
        "platform": first_platform,
        "niche": state.get("niche", ""),
        "deliverable_type": state.get("deliverable_type", "reel"),
        "qty": qty,
        "avg_views": avg_views,
        "final_price_brl": agreed_price,
        "cpm_brl": round(cpm, 2),
    }

    name = state.get("name") or "você"
    closing_text = DEAL_CLOSING_MESSAGE.format(name=name)

    return {
        "deal_to_save": deal,
        "current_node": "save_deal",
        "messages": [AIMessage(content=closing_text)],
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
    if state.get("deal_accepted"):
        return "save_deal"
    if state.get("approval_required"):
        return "approval"
    return END  # wait for user reply


def after_approval(state: NegotiatorState) -> str:
    if state.get("deal_accepted"):
        return "save_deal"
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
    graph.add_node("save_deal", save_deal_node)
    graph.add_node("close", close_node)

    graph.add_edge(START, "qualify")
    graph.add_conditional_edges("qualify", after_qualify)
    graph.add_edge("retrieve_benchmarks", "price")
    graph.add_edge("price", "negotiate")
    graph.add_conditional_edges("negotiate", after_negotiate)
    graph.add_conditional_edges("approval", after_approval)
    graph.add_edge("save_deal", END)
    graph.add_edge("close", END)

    return graph.compile(checkpointer=checkpointer)
