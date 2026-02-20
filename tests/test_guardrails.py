"""Tests for guardrails."""

import pytest

from app.tools.guardrails import (
    HANDOFF_SUFFIX,
    SENSITIVE_RESPONSE,
    append_handoff_suffix,
    check_human_handoff,
    check_sensitive_data,
)


class TestCheckHumanHandoff:
    def test_detects_humano(self):
        assert check_human_handoff("quero falar com humano") is True

    def test_detects_atendente(self):
        assert check_human_handoff("me passa pra um atendente") is True

    def test_detects_pessoa(self):
        assert check_human_handoff("quero falar com uma pessoa real") is True

    def test_no_handoff(self):
        assert check_human_handoff("quero negociar o preço") is False


class TestCheckSensitiveData:
    def test_detects_cpf(self):
        assert check_sensitive_data("meu cpf é 123.456.789-00") is True

    def test_detects_cpf_no_dots(self):
        assert check_sensitive_data("cpf 12345678900") is True

    def test_detects_card(self):
        assert check_sensitive_data("cartão 4111 1111 1111 1111") is True

    def test_detects_password(self):
        assert check_sensitive_data("senha: minhasenha123") is True

    def test_no_sensitive(self):
        assert check_sensitive_data("meu nome é João") is False


class TestAppendHandoffSuffix:
    def test_appends_suffix(self):
        result = append_handoff_suffix("Olá!")
        assert result.endswith(HANDOFF_SUFFIX)
        assert result.startswith("Olá!")
