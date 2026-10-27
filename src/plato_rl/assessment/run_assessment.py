"""Submit custom assessment job to Azure ML."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from plato_rl.azure.jobs import submit_and_monitor
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def render_job_yaml(
    config: PlatoConfig,
    output_dir: Path,
    checkpoint_path: str = "./outputs/basic",
    init_conditions_path: str = "aml_src/assessment/init_conditions.json",
) -> Path:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("job_assessment.yml.j2")

    aml_src = Path(config.project_root) / "aml_src" / "assessment"
    abs_checkpoint = str(Path(config.project_root) / checkpoint_path)
    rendered = template.render(
        code_path=str(aml_src),
        environment_name=config.environment_name,
        compute_name=config.compute_name,
        checkpoint_path=abs_checkpoint,
        init_conditions_path=str(Path(config.project_root) / init_conditions_path),
        instance_count=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    job_path = output_dir / "job_assessment.yml"
    job_path.write_text(rendered)
    return job_path


def run_assessment(config: PlatoConfig, checkpoint_path: str = "./outputs/basic") -> str:
    ui.info("Preparing custom assessment job...")
    output_dir = Path(config.project_root) / ".generated"
    job_path = render_job_yaml(config, output_dir, checkpoint_path)
    return submit_and_monitor(
        str(job_path), config, output_dir="./outputs/assessment"
    )
