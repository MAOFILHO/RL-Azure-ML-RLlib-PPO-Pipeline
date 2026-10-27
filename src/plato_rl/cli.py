"""Plato RL Azure — CLI entrypoint with all workflow commands."""
import shutil
import subprocess
import sys
from pathlib import Path

import click

from plato_rl import console as ui
from plato_rl.config import load_config


def _find_checkpoint(base_path: str) -> str:
    """Find the actual checkpoint directory containing algorithm_state.pkl."""
    base = Path(base_path)
    if (base / "algorithm_state.pkl").exists():
        return str(base)
    for pkl in sorted(base.rglob("algorithm_state.pkl")):
        return str(pkl.parent)
    return str(base)


@click.group()
@click.pass_context
def cli(ctx):
    """Plato RL Azure — Train and deploy RL agents on Azure ML."""
    ctx.ensure_object(dict)


@cli.command()
def check():
    """Step 0: Verify all prerequisites."""
    ui.step_header("0", "check", "Verifying prerequisites (Python, Azure CLI, az ml extension)")

    all_ok = True

    # Python version
    v = sys.version_info
    if v.major == 3 and v.minor == 10:
        ui.success(f"Python {v.major}.{v.minor}.{v.micro}")
    elif v.major == 3 and v.minor in (9, 11):
        ui.warn(f"Python {v.major}.{v.minor}.{v.micro} — Python 3.10 recommended for Ray 2.5.0")
    else:
        ui.error(f"Python {v.major}.{v.minor}.{v.micro} — Python 3.10 required")
        all_ok = False

    # Azure CLI
    from plato_rl.azure.auth import verify_prerequisites
    if not verify_prerequisites():
        all_ok = False

    # Config
    config = load_config()
    errors = config.validate()
    if errors:
        for e in errors:
            ui.warn(e)
        ui.warn("Create a .env file from .env.example and fill in your Azure details")
    else:
        ui.success("Configuration loaded from .env")

    if all_ok:
        ui.step_complete("0", "check")
        ui.next_step("0", "provision")
    else:
        ui.step_failed("0", "check", "Some prerequisites are missing")


@cli.command()
def provision():
    """Step 1: Create Azure resource group, workspace, and compute cluster."""
    ui.step_header("1", "provision", "Creating Azure resources (idempotent)")

    config = load_config()
    errors = config.validate()
    if errors:
        for e in errors:
            ui.error(e)
        return

    from plato_rl.azure.auth import verify_subscription
    if not verify_subscription(config):
        return

    # Render conda.yml from template
    from jinja2 import Environment, FileSystemLoader
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("conda.yml.j2")
    rendered = template.render()
    generated_dir = Path(config.project_root) / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    conda_path = generated_dir / "conda.yml"
    conda_path.write_text(rendered)

    from plato_rl.azure.provision import provision_all
    provision_all(config, str(conda_path))

    ui.step_complete("1", "provision")
    ui.next_step("1", "smoke-test")


@cli.command("setup-env")
def setup_env():
    """Step 2: Create AML conda environment."""
    ui.step_header("2", "setup-env", "Creating AML conda environment from template")

    config = load_config()

    from jinja2 import Environment, FileSystemLoader
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("conda.yml.j2")
    rendered = template.render()
    generated_dir = Path(config.project_root) / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    conda_path = generated_dir / "conda.yml"
    conda_path.write_text(rendered)
    ui.success(f"Conda environment file generated: {conda_path}")

    from plato_rl.azure.provision import ensure_environment
    ensure_environment(config, str(conda_path))

    ui.step_complete("2", "setup-env")
    ui.next_step("2", "smoke-test")


@cli.command("smoke-test")
def smoke_test():
    """Step 3: Run local PPO training on SimpleAdder (~30s)."""
    ui.step_header("3", "smoke-test", "Running local PPO training on SimpleAdder")

    from plato_rl.training.smoke_test import run_smoke_test
    try:
        run_smoke_test()
        ui.step_complete("3", "smoke-test")
        ui.next_step("3", "train")
    except Exception as e:
        ui.step_failed("3", "smoke-test", str(e))
        raise


@cli.command()
def train():
    """Step 4: Submit basic training job to Azure ML."""
    ui.step_header("4", "train", "Submitting basic training job to Azure ML")

    config = load_config()
    from plato_rl.training.basic_training import run_basic_training
    status = run_basic_training(config)

    if status == "Completed":
        ui.step_complete("4", "train")
        ui.next_step("4", "curriculum")
    else:
        ui.step_failed("4", "train", f"Job ended with status: {status}")


@cli.command()
def curriculum():
    """Step 5: Submit curriculum learning job to Azure ML."""
    ui.step_header("5", "curriculum", "Submitting curriculum learning job to Azure ML")

    config = load_config()
    from plato_rl.training.curriculum import run_curriculum
    status = run_curriculum(config)

    if status == "Completed":
        ui.step_complete("5", "curriculum")
        ui.next_step("5", "hpt")
    else:
        ui.step_failed("5", "curriculum", f"Job ended with status: {status}")


@cli.command()
def hpt():
    """Step 6: Submit hyperparameter tuning job with MLflow."""
    ui.step_header("6", "hpt", "Submitting hyperparameter tuning job with MLflow")

    config = load_config()
    from plato_rl.training.hyperparameter import run_hyperparameter_tuning
    status = run_hyperparameter_tuning(config)

    if status == "Completed":
        ui.info("View MLflow metrics: open the Azure ML Studio link above, then click 'Include child jobs'")
        ui.step_complete("6", "hpt")
        ui.next_step("6", "trajectories")
    else:
        ui.step_failed("6", "hpt", f"Job ended with status: {status}")


@cli.command()
def trajectories():
    """Step 7: Submit trajectory logging job to Azure ML."""
    ui.step_header("7", "trajectories", "Submitting trajectory logging job to Azure ML")

    config = load_config()
    from jinja2 import Environment, FileSystemLoader
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("job_trajectories.yml.j2")

    aml_src = Path(config.project_root) / "aml_src" / "trajectories"
    rendered = template.render(
        code_path=str(aml_src),
        environment_name=config.environment_name,
        compute_name=config.compute_name,
        instance_count=1,
    )

    generated_dir = Path(config.project_root) / ".generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    job_path = generated_dir / "job_trajectories.yml"
    job_path.write_text(rendered)

    from plato_rl.azure.jobs import submit_and_monitor
    status = submit_and_monitor(
        str(job_path), config, output_dir="./outputs/trajectories"
    )

    if status == "Completed":
        ui.step_complete("7", "trajectories")
        ui.next_step("7", "assess")
    else:
        ui.step_failed("7", "trajectories", f"Job ended with status: {status}")


@cli.command()
@click.option("--checkpoint", default="./outputs/basic", help="Path to checkpoint directory")
def assess(checkpoint):
    """Step 8: Submit custom assessment job to Azure ML."""
    ui.step_header("8", "assess", "Submitting custom assessment job to Azure ML")

    config = load_config()
    checkpoint_path = _find_checkpoint(checkpoint)
    ui.info(f"Using checkpoint: {checkpoint_path}")
    from plato_rl.assessment.run_assessment import run_assessment
    status = run_assessment(config, checkpoint_path=checkpoint_path)

    if status == "Completed":
        ui.step_complete("8", "assess")
        ui.next_step("8", "deploy")
    else:
        ui.step_failed("8", "assess", f"Job ended with status: {status}")


@cli.command()
@click.option("--checkpoint", default="./outputs/basic", help="Path to checkpoint directory")
def deploy(checkpoint):
    """Step 9: Deploy trained agent to AML managed endpoint."""
    ui.step_header("9", "deploy", "Deploying trained agent to Azure ML managed endpoint")

    config = load_config()
    checkpoint_path = _find_checkpoint(checkpoint)
    ui.info(f"Using checkpoint: {checkpoint_path}")
    from plato_rl.deploy.aml_endpoint import deploy_agent
    try:
        deploy_agent(config, checkpoint_path)
        ui.step_complete("9", "deploy")
        ui.next_step("9", "teardown")
    except Exception as e:
        ui.step_failed("9", "deploy", str(e))


@cli.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def teardown(force):
    """Step 10: Delete all Azure resources and verify cleanup."""
    ui.step_header("10", "teardown", "Deleting all Azure resources and verifying cleanup")

    config = load_config()
    from plato_rl.azure.teardown import teardown as do_teardown
    ok = do_teardown(config, force=force)

    if ok:
        ui.step_complete("10", "teardown")
    else:
        ui.step_failed("10", "teardown", "Teardown incomplete — check output above")


@cli.command("run-all")
@click.option("--skip-azure", is_flag=True, help="Only run local steps (0, 3)")
def run_all(skip_azure):
    """Run complete workflow (Steps 0-9)."""
    ui.console.print("\n[bold magenta]═══ Plato RL Azure — Full Pipeline ═══[/bold magenta]\n")

    ctx = click.Context(cli)

    steps = [
        ("check", check),
        ("smoke-test", smoke_test),
    ]

    if not skip_azure:
        steps += [
            ("provision", provision),
            ("train", train),
            ("curriculum", curriculum),
            ("hpt", hpt),
            ("trajectories", trajectories),
        ]

    for name, cmd in steps:
        try:
            ctx.invoke(cmd)
        except (click.Abort, SystemExit):
            ui.error(f"Pipeline stopped at step: {name}")
            return
        except Exception as e:
            ui.error(f"Pipeline failed at step {name}: {e}")
            return

    ui.console.print("\n[bold green]═══ Pipeline Complete ═══[/bold green]")
    ui.show_workflow_status({"0", "1", "2", "3", "4", "5", "6", "7"})


@cli.command()
def status():
    """Show workflow status overview."""
    ui.show_workflow_status(set())
    ui.info("Run individual steps with: plato <step-name>")
    ui.info("Run full pipeline with: plato run-all")


def main():
    cli()


if __name__ == "__main__":
    main()
