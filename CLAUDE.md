# CLAUDE.md

## O que e este projeto

MVP de um agente negociador de influenciadores para a Gocase. Conversa com influenciadores via terminal, negocia precos de conteudo (reels, posts, stories, videos) usando benchmarks de mercado e persiste tudo em SQLite.

## Stack

- **Python 3.11+**
- **LangGraph** — state machine de negociacao (qualify → retrieve_benchmarks → price → negotiate → approval → save_deal → close)
- **OpenAI Responses API** (gpt-4o-mini) — LLM + function calling
- **SQLAlchemy 2.0** + **SQLite** — persistencia (negotiator.db + checkpoints.sqlite)
- **Typer + Rich** — CLI
- **Pydantic** — validacao

## Estrutura

```
app/
  agents/negotiator.py   # Grafo LangGraph principal (~850 linhas)
  agents/state.py        # TypedDict do estado
  core/orchestrator.py   # Ponte CLI <-> LangGraph
  core/registry.py       # Registro de agentes
  core/store.py          # CRUD do banco
  tools/__init__.py      # Schemas de tools OpenAI
  tools/pricing.py       # Calculo de faixa de preco
  tools/retrieval.py     # Consulta benchmarks
  tools/guardrails.py    # Checagens de seguranca
  db/models.py           # Modelos SQLAlchemy
  db/session.py          # Engine e session factory
  db/seed.py             # Dados de exemplo (20 deals)
  cli.py                 # Interface Typer
  __main__.py            # Entry point
tests/
data/                    # SQLite databases (gitignored em producao)
```

## Comandos uteis

```bash
# Instalar
pip install -e ".[dev]"

# Seed do banco
python -m app seed

# Iniciar negociacao
python -m app chat --agent negotiator --influencer "+5585999999999" --new

# Retomar conversa
python -m app chat --agent negotiator --influencer "+5585999999999"

# Listar conversas
python -m app list-conversations

# Testes
pytest tests/ -v
```

## Variaveis de ambiente

Copiar `.env.example` para `.env` e preencher:
- `OPENAI_API_KEY` — chave da OpenAI
- `OPENAI_MODEL` — modelo (default: gpt-4o-mini)
- `DATABASE_URL` — SQLite path (default: sqlite:///data/negotiator.db)
- `CHECKPOINT_DB` — checkpoints LangGraph (default: data/checkpoints.sqlite)

## Conceitos-chave

- **Persona "Raimunda"**: agente negociadora da Gocase, prompts em portugues BR
- **Faixa de preco**: floor (70% do target), target, ceiling (130% do target). Formula: `(avg_views * qty * cpm) / 1000`
- **Approval workflow**: precos fora da faixa requerem aprovacao humana via interrupt
- **Human handoff**: keywords como "humano", "atendente" transferem para operador
- **Multi-plataforma**: influenciador pode ter instagram, tiktok, youtube (comma-separated set)
- **Coleta pos-deal**: apos fechar, coleta email, CPF, endereco, modelo do celular
- **Benchmarks**: deals historicos ordenados por proximidade de views, nao por preco

## Convencoes de codigo

- Todo o sistema (prompts, UI, mensagens) esta em **portugues BR**
- Nomes de variaveis e codigo em **ingles**
- Tool dispatch centralizado em `_dispatch_tool()` no negotiator.py
- Estado do grafo usa TypedDict (nao Pydantic) com Annotated list para messages
- Perfil do influenciador e acumulado incrementalmente — nunca sobrescreve campos existentes
- Preco extraido via regex `r"R\$\s?([\d.,]+)"` pegando ultimo match da resposta
