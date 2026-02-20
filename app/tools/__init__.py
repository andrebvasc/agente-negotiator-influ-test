"""OpenAI function-calling tool schemas."""

OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "retrieve_benchmarks",
        "description": (
            "Busca deals históricos similares no banco de dados para embasar a negociação. "
            "Retorna estatísticas (count, avg_cpm, median_price, min, max) e amostras."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "Plataforma do influenciador (instagram, tiktok, youtube)",
                },
                "niche": {
                    "type": "string",
                    "description": "Nicho do influenciador",
                },
                "deliverable_type": {
                    "type": "string",
                    "description": "Tipo de entregável (reel, post, story, video)",
                },
                "avg_views": {
                    "type": "integer",
                    "description": "Média de visualizações do influenciador",
                },
                "k": {
                    "type": "integer",
                    "description": "Número de amostras a retornar (default 5)",
                },
            },
            "required": ["platform", "deliverable_type", "avg_views"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "calculate_price_range",
        "description": (
            "Calcula faixa de preço (floor/target/ceiling) com base em views, "
            "quantidade, CPM alvo e benchmarks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "avg_views": {
                    "type": "integer",
                    "description": "Média de visualizações",
                },
                "qty": {
                    "type": "integer",
                    "description": "Quantidade de peças",
                },
                "target_cpm_brl": {
                    "type": "number",
                    "description": "CPM alvo em reais",
                },
                "benchmarks": {
                    "type": "object",
                    "description": "Estatísticas retornadas por retrieve_benchmarks",
                },
            },
            "required": ["avg_views", "qty", "target_cpm_brl"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_approval_required",
        "description": (
            "Verifica se a proposta precisa de aprovação humana. "
            "Retorna true se o preço proposto está fora da faixa ou não há benchmarks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "proposed_brl": {
                    "type": "number",
                    "description": "Valor proposto em reais",
                },
                "price_range": {
                    "type": "object",
                    "description": "Faixa de preço {floor, target, ceiling}",
                    "properties": {
                        "floor": {"type": "number"},
                        "target": {"type": "number"},
                        "ceiling": {"type": "number"},
                    },
                },
                "benchmarks": {
                    "type": "object",
                    "description": "Estatísticas de benchmark",
                },
            },
            "required": ["proposed_brl", "price_range"],
            "additionalProperties": False,
        },
    },
]
