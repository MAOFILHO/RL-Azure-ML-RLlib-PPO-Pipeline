from plato_rl.azure.cli_runner import (
    AzureCliError,
    check_az_installed,
    check_az_ml_extension,
    get_current_subscription,
    run_az,
)
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def verify_prerequisites() -> bool:
    all_ok = True

    if not check_az_installed():
        ui.error("Azure CLI not found. Install: https://aka.ms/installazureclimacos")
        return False
    ui.success("Azure CLI installed")

    if not check_az_ml_extension():
        ui.warn("Azure ML extension not found. Installing...")
        try:
            run_az("extension", "add", "-n", "ml", "--yes", parse_json=False)
            ui.success("Azure ML extension installed")
        except AzureCliError as e:
            ui.error(f"Failed to install ML extension: {e}")
            return False
    else:
        ui.success("Azure ML extension installed")

    sub = get_current_subscription()
    if sub:
        ui.success(f"Logged in to Azure (subscription: {sub[:8]}...)")
    else:
        ui.error("Not logged in to Azure. Run: az login")
        all_ok = False

    return all_ok


def verify_subscription(config: PlatoConfig) -> bool:
    current = get_current_subscription()
    if not current:
        ui.error("Not logged in to Azure. Run: az login")
        return False

    if config.subscription_id and current != config.subscription_id:
        ui.info(f"Switching to subscription {config.subscription_id[:8]}...")
        try:
            run_az(
                "account", "set",
                "--subscription", config.subscription_id,
                parse_json=False,
            )
            ui.success("Subscription set")
        except AzureCliError as e:
            ui.error(f"Failed to set subscription: {e}")
            return False

    return True
