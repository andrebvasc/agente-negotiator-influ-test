"""Guardrails: human handoff detection and sensitive data checks."""

import re

HANDOFF_PATTERNS = re.compile(
    r"\b(humano|pessoa|atendente|operador|supervisor|gerente)\b",
    re.IGNORECASE,
)

SENSITIVE_PATTERNS = re.compile(
    r"(\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b)"  # CPF
    r"|(\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b)"  # cartão
    r"|(\bsenha\s*[:=]\s*\S+)",  # senha
    re.IGNORECASE,
)

HANDOFF_SUFFIX = "\n\n_Se preferir falar com uma pessoa, digite **HUMANO**._"

SENSITIVE_RESPONSE = (
    "Por segurança, não posso coletar dados sensíveis como CPF, "
    "número de cartão ou senhas. Por favor, nunca compartilhe essas "
    "informações por este canal."
)


def check_human_handoff(text: str) -> bool:
    """Return True if user is requesting a human agent."""
    return bool(HANDOFF_PATTERNS.search(text))


def check_sensitive_data(text: str) -> bool:
    """Return True if text contains sensitive data (CPF, card number, password)."""
    return bool(SENSITIVE_PATTERNS.search(text))


def append_handoff_suffix(response: str) -> str:
    """Append the human-handoff reminder to a response (if not already present)."""
    if HANDOFF_SUFFIX.strip() in response:
        return response
    return response + HANDOFF_SUFFIX
