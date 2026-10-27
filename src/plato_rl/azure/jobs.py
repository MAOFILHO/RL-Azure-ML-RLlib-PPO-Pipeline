import time
from pathlib import Path

from plato_rl.azure.cli_runner import AzureCliError, run_az
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def get_studio_url_from_job(job_result: dict) -> str:
    """Extract the Azure ML Studio URL from the job creation response."""
    return job_result.get("services", {}).get("Studio", {}).get("endpoint", "")


def submit_job(job_yml_path: str, config: PlatoConfig) -> str:
    ui.info(f"Submitting job from {job_yml_path}...")
    result = run_az(
        "ml", "job", "create",
        "-f", job_yml_path,
        "--resource-group", config.resource_group,
        "--workspace-name", config.workspace_name,
    )
    if isinstance(result, dict):
        job_name = result.get("name", "unknown")
        display_name = result.get("display_name", job_name)
        ui.success(f"Job '{display_name}' submitted (name: {job_name})")
        studio_url = get_studio_url_from_job(result)
        if studio_url:
            ui.info(f"Azure ML Studio: {studio_url}")
        return job_name
    raise AzureCliError("az ml job create", -1, "Unexpected response format")


def poll_job(
    job_name: str,
    config: PlatoConfig,
    poll_interval: int = 30,
    timeout: int = 3600,
) -> str:
    ui.info(f"Monitoring job '{job_name}' (timeout: {timeout // 60} min)...")
    start = time.time()
    last_status = ""

    while time.time() - start < timeout:
        try:
            result = run_az(
                "ml", "job", "show",
                "--name", job_name,
                "--resource-group", config.resource_group,
                "--workspace-name", config.workspace_name,
            )
            if isinstance(result, dict):
                status = result.get("status", "Unknown")
                if status != last_status:
                    elapsed = int(time.time() - start)
                    ui.info(f"Job status: {status} (elapsed: {elapsed}s)")
                    last_status = status

                if status in ("Completed", "Finished"):
                    ui.success(f"Job '{job_name}' completed successfully!")
                    return "Completed"
                elif status in ("Failed", "Canceled", "CancelRequested"):
                    ui.error(f"Job '{job_name}' {status.lower()}")
                    return status
        except AzureCliError as e:
            ui.warn(f"Error polling job: {e}")

        time.sleep(poll_interval)

    ui.error(f"Job '{job_name}' timed out after {timeout}s")
    return "Timeout"


def download_outputs(
    job_name: str,
    config: PlatoConfig,
    output_dir: str = "./outputs",
) -> Path:
    ui.info(f"Downloading job outputs to {output_dir}...")
    try:
        run_az(
            "ml", "job", "download",
            "--name", job_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--download-path", output_dir,
            parse_json=False,
            timeout=300,
        )
        ui.success(f"Outputs downloaded to {output_dir}")
    except AzureCliError as e:
        ui.warn(f"Could not download outputs: {e}")
    return Path(output_dir)


def submit_and_monitor(
    job_yml_path: str,
    config: PlatoConfig,
    download: bool = True,
    output_dir: str = "./outputs",
    timeout: int = 3600,
) -> str:
    job_name = submit_job(job_yml_path, config)
    status = poll_job(job_name, config, timeout=timeout)

    if status == "Completed" and download:
        download_outputs(job_name, config, output_dir)

    return status
