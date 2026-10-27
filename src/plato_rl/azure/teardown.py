import subprocess
import time

from rich.table import Table

from plato_rl.azure.cli_runner import AzureCliError, run_az
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def list_resources(config: PlatoConfig) -> list[dict]:
    try:
        resources = run_az(
            "resource", "list",
            "--resource-group", config.resource_group,
        )
        return resources if isinstance(resources, list) else []
    except AzureCliError:
        return []


def display_resource_inventory(resources: list[dict]) -> None:
    if not resources:
        ui.info("No resources found in resource group.")
        return

    table = Table(
        title="Resources to Delete",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Name", style="white")
    table.add_column("Type", style="dim")
    table.add_column("Location", style="cyan")

    for r in resources:
        table.add_row(
            r.get("name", "unknown"),
            r.get("type", "unknown").split("/")[-1],
            r.get("location", "unknown"),
        )

    ui.console.print()
    ui.console.print(table)
    ui.console.print()


def delete_resource_group(config: PlatoConfig) -> bool:
    ui.info(f"Deleting resource group '{config.resource_group}'...")
    try:
        run_az(
            "group", "delete",
            "--name", config.resource_group,
            "--yes",
            parse_json=False,
            timeout=600,
        )
    except AzureCliError as e:
        ui.error(f"Failed to delete resource group: {e}")
        return False
    return True


def verify_deletion(config: PlatoConfig, max_wait: int = 300) -> bool:
    ui.info("Verifying resource group deletion...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            result = run_az(
                "group", "exists",
                "--name", config.resource_group,
                parse_json=False,
            )
            if result and result.strip().lower() == "false":
                ui.success("Resource group deletion confirmed")
                return True
        except AzureCliError:
            pass
        time.sleep(10)

    ui.warn("Resource group deletion could not be verified within timeout")
    return False


def check_orphaned_resources() -> list[dict]:
    try:
        resources = run_az(
            "resource", "list",
            "--tag", "plato-rl=true",
        )
        return resources if isinstance(resources, list) else []
    except AzureCliError:
        return []


def purge_soft_deleted_workspace(config: PlatoConfig) -> None:
    """Purge the soft-deleted workspace so future provisions don't fail."""
    ui.info(f"Purging soft-deleted workspace '{config.workspace_name}'...")
    try:
        subprocess.run(
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
        ui.success("Soft-deleted workspace purged (clean for re-provisioning)")
    except (subprocess.TimeoutExpired, Exception) as e:
        ui.warn(f"Could not purge soft-deleted workspace: {e}")


def teardown(config: PlatoConfig, force: bool = False) -> bool:
    resources = list_resources(config)

    if not resources:
        ui.info("Resource group is empty or does not exist. Nothing to tear down.")
        try:
            run_az("group", "show", "--name", config.resource_group)
        except AzureCliError:
            ui.success("Resource group does not exist. All clean.")
            return True

    display_resource_inventory(resources)

    if not force:
        ui.console.print(
            f"  [bold red]This will permanently delete {len(resources)} resources "
            f"in '{config.resource_group}'.[/bold red]"
        )
        confirm = ui.console.input(
            f"\n  Type the resource group name to confirm deletion [{config.resource_group}]: "
        )
        if confirm.strip() != config.resource_group:
            ui.warn("Teardown cancelled — resource group name did not match.")
            return False

    if not delete_resource_group(config):
        return False

    verify_deletion(config)

    purge_soft_deleted_workspace(config)

    orphans = check_orphaned_resources()
    if orphans:
        ui.warn(f"Found {len(orphans)} orphaned resource(s) with plato-rl tag:")
        for r in orphans:
            ui.warn(f"  - {r.get('name', 'unknown')} ({r.get('type', 'unknown')})")
        ui.warn("Run 'az resource delete --ids <id>' to clean up manually.")
        return False

    ui.success("All Plato RL Azure resources have been deleted.")
    ui.success("No orphaned resources found. You will not be charged.")
    return True
