"""CLI interface using Typer + Rich."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.orchestrator import Orchestrator
from app.db.seed import seed as seed_db
from app.db.session import SessionLocal, init_db

app = typer.Typer(help="Agente Negociador de Influenciadores")
console = Console()


@app.command()
def chat(
    agent: str = typer.Option("negotiator", help="ID do agente"),
    influencer: str = typer.Option(..., help="Telefone do influenciador"),
    new: bool = typer.Option(False, help="Forçar nova conversa"),
):
    """Iniciar chat de negociação com um influenciador."""
    console.print(
        Panel(
            "[bold green]Agente Negociador de Influenciadores[/bold green]\n"
            f"Agente: {agent} | Influenciador: {influencer}",
            title="Bem-vindo",
        )
    )

    orchestrator = Orchestrator(agent_id=agent)

    try:
        result = orchestrator.start_or_resume_conversation(influencer, new=new)
        thread_id = result["thread_id"]
        conv = result["conversation"]

        if result["resumed"]:
            console.print("[dim]Conversa anterior retomada.[/dim]")
        else:
            console.print("[dim]Nova conversa iniciada.[/dim]")

        console.print(
            "[dim]Digite sua mensagem como influenciador. "
            "Ctrl+C ou 'sair' para encerrar.[/dim]\n"
        )

        while True:
            try:
                user_input = Prompt.ask("[bold cyan]Você[/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Encerrando...[/dim]")
                break

            if user_input.strip().lower() in ("sair", "exit", "quit"):
                console.print("[dim]Encerrando conversa.[/dim]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold yellow]Pensando...[/bold yellow]"):
                response = orchestrator.process_message(
                    thread_id, conv.id, user_input
                )

            console.print(
                Panel(
                    response["response"],
                    title="[bold magenta]Negociador[/bold magenta]",
                    border_style="magenta",
                )
            )

            if response["owner"] == "human":
                console.print(
                    "[bold red]Conversa transferida para atendente humano.[/bold red]"
                )
                break

            if response["approval_required"]:
                console.print(
                    Panel(
                        "[bold yellow]APROVAÇÃO NECESSÁRIA[/bold yellow]\n"
                        "A proposta atual requer aprovação do operador.",
                        border_style="yellow",
                    )
                )
                decision = Prompt.ask(
                    "Decisão (sim/não/valor para contraproposta)"
                )
                if decision.replace(".", "").replace(",", "").isdigit():
                    counter = float(decision.replace(",", "."))
                    approval = orchestrator.handle_approval(
                        thread_id,
                        {"approved": False, "counter_offer_brl": counter},
                    )
                elif decision.lower() in ("sim", "s", "yes"):
                    approval = orchestrator.handle_approval(
                        thread_id, {"approved": True}
                    )
                else:
                    approval = orchestrator.handle_approval(
                        thread_id, {"approved": False}
                    )

                console.print(
                    Panel(
                        approval["response"],
                        title="[bold magenta]Negociador[/bold magenta]",
                        border_style="magenta",
                    )
                )

    finally:
        orchestrator.close()


@app.command()
def seed():
    """Popular banco com deals fictícios."""
    init_db()
    count = seed_db()
    if count:
        console.print(f"[green]{count} deals inseridos com sucesso![/green]")
    else:
        console.print("[yellow]Banco já contém dados. Nenhum deal inserido.[/yellow]")


@app.command(name="list-conversations")
def list_conversations():
    """Listar conversas existentes."""
    init_db()
    session = SessionLocal()
    try:
        from app.db.models import Conversation

        convs = (
            session.query(Conversation)
            .order_by(Conversation.created_at.desc())
            .all()
        )
        if not convs:
            console.print("[dim]Nenhuma conversa encontrada.[/dim]")
            return

        for c in convs:
            console.print(
                f"[bold]{c.thread_id}[/bold] | "
                f"Status: {c.status} | Owner: {c.owner} | "
                f"Criada: {c.created_at}"
            )
    finally:
        session.close()
