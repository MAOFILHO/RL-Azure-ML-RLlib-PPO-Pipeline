from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

STEPS = [
    ("0", "check", "Verify prerequisites (Python, Azure CLI, az ml extension)"),
    ("1", "provision", "Create Azure resource group, workspace, and compute cluster"),
    ("2", "setup-env", "Create AML conda environment from template"),
    ("3", "smoke-test", "Run local PPO training on SimpleAdder (~30s)"),
    ("4", "train", "Submit basic training job to Azure ML"),
    ("5", "curriculum", "Submit curriculum learning job to Azure ML"),
    ("6", "hpt", "Submit hyperparameter tuning job with MLflow"),
    ("7", "trajectories", "Submit trajectory logging job to Azure ML"),
    ("8", "assess", "Submit custom assessment job to Azure ML"),
    ("9", "deploy", "Deploy trained agent to AML managed endpoint"),
    ("10", "teardown", "Delete all Azure resources and verify cleanup"),
]


def step_header(step_num: str, title: str, description: str) -> None:
    panel = Panel(
        Text(description, style="white"),
        title=f"[bold cyan]Step {step_num}: {title}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def step_complete(step_num: str, title: str) -> None:
    console.print(
        f"  [bold green]✓[/bold green] Step {step_num} ({title}) completed successfully.\n"
    )


def step_failed(step_num: str, title: str, error: str) -> None:
    console.print(
        f"  [bold red]✗[/bold red] Step {step_num} ({title}) failed: {error}\n"
    )


def show_workflow_status(completed: set[str]) -> None:
    table = Table(
        title="Plato RL Azure — Workflow Status",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Command", style="white", width=16)
    table.add_column("Description", style="dim")
    table.add_column("Status", justify="center", width=10)

    for num, cmd, desc in STEPS:
        if num in completed:
            status = "[bold green]✓ Done[/bold green]"
        else:
            status = "[dim]• Pending[/dim]"
        table.add_row(num, f"plato {cmd}", desc, status)

    console.print()
    console.print(table)
    console.print()


def info(msg: str) -> None:
    console.print(f"  [blue]ℹ[/blue]  {msg}")


def success(msg: str) -> None:
    console.print(f"  [green]✓[/green]  {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/red]  {msg}")


def cost_warning() -> None:
    console.print(
        Panel(
            "[yellow]Azure resources incur costs while active.\n"
            "Run [bold]plato teardown[/bold] when done to delete all resources.\n"
            "Compute cluster min_nodes=0 means nodes deallocate when idle.[/yellow]",
            title="[bold yellow]Cost Warning[/bold yellow]",
            border_style="yellow",
        )
    )


def next_step(current: str, next_cmd: Optional[str] = None) -> None:
    if next_cmd:
        console.print(f"\n  [dim]Next step:[/dim] [bold]plato {next_cmd}[/bold]\n")
