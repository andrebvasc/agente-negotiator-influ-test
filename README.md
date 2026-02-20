# Agente Negociador de Influenciadores — MVP

MVP local de um agente negociador que interage com influenciadores via terminal. Usa LangGraph para state machine, OpenAI Responses API para LLM + tool calling e SQLite para persistência.

## Setup

```bash
# 1. Instalar dependências
pip install -e ".[dev]"

# 2. Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com sua OPENAI_API_KEY

# 3. Popular banco com deals de exemplo
python -m app seed
```

## Uso

```bash
# Iniciar negociação
python -m app chat --agent negotiator --influencer "+5585999999999" --new

# Retomar conversa existente
python -m app chat --agent negotiator --influencer "+5585999999999"

# Listar conversas
python -m app list-conversations
```

## Testes

```bash
pytest tests/ -v
```

## Arquitetura

- **CLI** (`app/cli.py`): REPL com Typer + Rich
- **Orchestrator** (`app/core/orchestrator.py`): ponte CLI ↔ LangGraph
- **LangGraph** (`app/agents/negotiator.py`): grafo com nós qualify → retrieve_benchmarks → price → negotiate → approval → close
- **Tools** (`app/tools/`): pricing, retrieval (benchmarks), guardrails
- **DB** (`app/db/`): SQLAlchemy 2.0 com SQLite

## Fluxo

1. **Qualificação**: coleta platform, deliverable_type, avg_views, qty, deadline
2. **Benchmarks**: busca deals históricos similares
3. **Precificação**: calcula faixa (floor 70% / target / ceiling 130%)
4. **Negociação**: LLM negocia usando dados e tools
5. **Aprovação**: interrupt quando proposta fora da faixa
6. **Fechamento**: gera resumo do acordo
