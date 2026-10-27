"""Submit hyperparameter tuning job to Azure ML."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from plato_rl.azure.jobs import submit_and_monitor
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def render_job_yaml(config: PlatoConfig, output_dir: Path) -> Path:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("job_hpt.yml.j2")

    aml_src = Path(config.project_root) / "aml_src" / "hpt"
    rendered = template.render(
        code_path=str(aml_src),
        environment_name=config.environment_name,
        compute_name=config.compute_name,
        num_tune_samples=config.num_tune_samples,
        instance_count=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "job_hpt.yml"
    job_path.write_text(rendered)
    return job_path


def run_hyperparameter_tuning(config: PlatoConfig) -> str:
    ui.info("Preparing hyperparameter tuning job...")
    output_dir = Path(config.project_root) / ".generated"
    job_path = render_job_yaml(config, output_dir)
    return submit_and_monitor(
        str(job_path), config, output_dir="./outputs/hpt", timeout=7200
    )
