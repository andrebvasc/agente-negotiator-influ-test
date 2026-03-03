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
- **LangGraph** (`app/agents/negotiator.py`): grafo com nós qualify → retrieve_benchmarks → price → negotiate → approval → save_deal → close
- **Tools** (`app/tools/`): pricing, retrieval (benchmarks), guardrails
- **DB** (`app/db/`): SQLAlchemy 2.0 com SQLite

## Fluxo

1. **Qualificação**: coleta platform, deliverable_type, avg_views, qty, deadline, niche, nome
2. **Benchmarks**: busca deals históricos similares (ordenados por proximidade de views)
3. **Precificação**: calcula faixa (floor 70% / target / ceiling 130%)
4. **Negociação**: LLM negocia usando dados internos e benchmarks de mercado
5. **Aprovação**: interrupt quando proposta fora da faixa (requer decisão humana)
6. **Salvamento**: persiste deal no banco com preço final e CPM
7. **Coleta pós-deal**: solicita dados pessoais (email, CPF, endereço, modelo do celular)

## Features

- **Multi-plataforma**: suporte a Instagram, TikTok e YouTube (influenciador pode atuar em várias)
- **Greeting system**: saudação personalizada para contatos novos e retornantes
- **Persistência de perfil**: dados do influenciador acumulados incrementalmente entre conversas
- **Histórico de contexto**: últimas 20 mensagens carregadas para continuidade
- **Salvamento de deals**: deals fechados persistidos com cálculo de CPM
- **Human handoff**: transferência para operador humano por keyword
- **Guardrails**: detecção de dados sensíveis (cartões, senhas)
- **Retomada de conversa**: checkpoints LangGraph permitem pausar e retomar negociações
