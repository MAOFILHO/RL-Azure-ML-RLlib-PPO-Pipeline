from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "plato_rl" / "templates"


def _render(template_name, **kwargs):
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template(template_name)
    return template.render(**kwargs)


def test_conda_template():
    rendered = _render("conda.yml.j2")
    assert "gymnasium==0.26.3" in rendered
    assert "ray[data,rllib]==2.5.0" in rendered
    assert "torch==2.0.1" in rendered


def test_job_basic_template():
    rendered = _render(
        "job_basic.yml.j2",
        code_path="./src",
        environment_name="test-env",
        compute_name="test-compute",
        instance_count=1,
    )
    assert "azureml:test-env@latest" in rendered
    assert "azureml:test-compute" in rendered
    assert "instance_count: 1" in rendered
    assert "type: mpi" in rendered


def test_job_hpt_template():
    rendered = _render(
        "job_hpt.yml.j2",
        code_path="./src",
        environment_name="test-env",
        compute_name="test-compute",
        num_tune_samples=5,
        instance_count=2,
    )
    assert "--num-tune-samples 5" in rendered
    assert "instance_count: 2" in rendered


def test_job_trajectories_template():
    rendered = _render(
        "job_trajectories.yml.j2",
        code_path="./src",
        environment_name="test-env",
        compute_name="test-compute",
        instance_count=1,
    )
    assert "workspaceblobstore" in rendered
    assert "trajectories" in rendered
