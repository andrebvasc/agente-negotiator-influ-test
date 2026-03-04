"""Guardrails: detecção de handoff humano e checagem de dados sensíveis."""

import re

HANDOFF_PATTERNS = re.compile(
    r"\b(humano|pessoa|atendente|operador|supervisor|gerente)\b",
    re.IGNORECASE,
)

SENSITIVE_PATTERNS = re.compile(
    r"(\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b)"  # cartão
    r"|(\bsenha\s*[:=]\s*\S+)",  # senha
    re.IGNORECASE,
)

HANDOFF_SUFFIX = "\n\n_Se preferir falar com uma pessoa, digite **HUMANO**._"

SENSITIVE_RESPONSE = (
    "Por segurança, não posso coletar dados como "
    "número de cartão de crédito ou senhas. Por favor, nunca compartilhe essas "
    "informações por este canal."
)


def check_human_handoff(text: str) -> bool:
    """Retorna True se o usuário está pedindo um atendente humano."""
    return bool(HANDOFF_PATTERNS.search(text))


def check_sensitive_data(text: str) -> bool:
    """Retorna True se o texto contém dados sensíveis (cartão de crédito, senha)."""
    return bool(SENSITIVE_PATTERNS.search(text))


def append_handoff_suffix(response: str) -> str:
    """Adiciona o lembrete de handoff humano à resposta (se ainda não presente)."""
    if HANDOFF_SUFFIX.strip() in response:
        return response
    return response + HANDOFF_SUFFIX
