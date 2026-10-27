import subprocess
import time

from plato_rl.azure.cli_runner import AzureCliError, run_az
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def purge_soft_deleted_workspace(config: PlatoConfig) -> bool:
    """Purge a soft-deleted workspace. Returns True if purge was attempted."""
    ui.info(f"Purging soft-deleted workspace '{config.workspace_name}'...")
    try:
        result = subprocess.run(
            [
                "az", "rest", "--method", "DELETE",
                "--url",
                f"https://management.azure.com/subscriptions/{config.subscription_id}"
                f"/resourceGroups/{config.resource_group}"
                f"/providers/Microsoft.MachineLearningServices"
                f"/workspaces/{config.workspace_name}"
                f"?api-version=2024-10-01&forcePurge=true",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            ui.success("Soft-deleted workspace purged")
            ui.info("Waiting 30s for Azure to propagate purge...")
            time.sleep(30)
            return True
        ui.warn(f"Purge returned rc={result.returncode}: {result.stderr.strip()}")
    except (subprocess.TimeoutExpired, Exception) as e:
        ui.warn(f"Could not purge soft-deleted workspace: {e}")
    return False


def ensure_resource_group(config: PlatoConfig) -> None:
    try:
        run_az("group", "show", "--name", config.resource_group)
        ui.success(f"Resource group '{config.resource_group}' already exists")
    except AzureCliError:
        ui.info(f"Creating resource group '{config.resource_group}' in {config.location}...")
        run_az(
            "group", "create",
            "--name", config.resource_group,
            "--location", config.location,
            "--tags", "plato-rl=true", "project=plato-rl-azure",
        )
        ui.success(f"Resource group '{config.resource_group}' created")


def ensure_workspace(config: PlatoConfig) -> None:
    try:
        run_az(
            "ml", "workspace", "show",
            "--name", config.workspace_name,
            "--resource-group", config.resource_group,
        )
        ui.success(f"Workspace '{config.workspace_name}' already exists")
        return
    except AzureCliError:
        pass

    purge_soft_deleted_workspace(config)

    ui.info(f"Creating workspace '{config.workspace_name}' (this may take 2-3 minutes)...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            run_az(
                "ml", "workspace", "create",
                "--name", config.workspace_name,
                "--resource-group", config.resource_group,
                "--location", config.location,
                "--tags", "plato-rl=true",
                timeout=300,
            )
            ui.success(f"Workspace '{config.workspace_name}' created")
            return
        except AzureCliError as e:
            if "soft-delete" in str(e).lower() or "Soft-deleted" in str(e):
                if attempt < max_retries - 1:
                    ui.warn("Workspace still soft-deleted. Waiting 60s before retry...")
                    time.sleep(60)
                    purge_soft_deleted_workspace(config)
                    continue
            raise


def ensure_compute(config: PlatoConfig) -> None:
    try:
        run_az(
            "ml", "compute", "show",
            "--name", config.compute_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
        )
        ui.success(f"Compute cluster '{config.compute_name}' already exists")
    except AzureCliError:
        ui.info(
            f"Creating compute cluster '{config.compute_name}' "
            f"(VM: {config.compute_vm_size}, nodes: {config.compute_min_nodes}-{config.compute_max_nodes})..."
        )
        run_az(
            "ml", "compute", "create",
            "--name", config.compute_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--type", "AmlCompute",
            "--size", config.compute_vm_size,
            "--min-instances", str(config.compute_min_nodes),
            "--max-instances", str(config.compute_max_nodes),
            "--tags", "plato-rl=true",
            timeout=120,
        )
        ui.success(f"Compute cluster '{config.compute_name}' created")
    ui.cost_warning()


def ensure_environment(config: PlatoConfig, conda_path: str) -> None:
    try:
        run_az(
            "ml", "environment", "show",
            "--name", config.environment_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--version", "1",
        )
        ui.success(f"Environment '{config.environment_name}' already exists")
    except AzureCliError:
        ui.info(
            f"Creating AML environment '{config.environment_name}' "
            f"(this may take 10-15 minutes on first build)..."
        )
        run_az(
            "ml", "environment", "create",
            "--name", config.environment_name,
            "--conda-file", conda_path,
            "--image", "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--tags", "plato-rl=true",
            timeout=600,
        )
        ui.success(f"Environment '{config.environment_name}' created")


def provision_all(config: PlatoConfig, conda_path: str) -> None:
    ensure_resource_group(config)
    ensure_workspace(config)
    ensure_compute(config)
    ensure_environment(config, conda_path)
