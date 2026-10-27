"""Deploy trained agent to AML managed online endpoint."""
import json
import tempfile
from pathlib import Path

from plato_rl.azure.cli_runner import AzureCliError, run_az
from plato_rl.config import PlatoConfig
from plato_rl import console as ui


def register_model(config: PlatoConfig, checkpoint_path: str) -> str:
    model_name = "plato-rl-agent"
    ui.info(f"Registering model '{model_name}' from {checkpoint_path}...")
    try:
        result = run_az(
            "ml", "model", "create",
            "--name", model_name,
            "--path", checkpoint_path,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--type", "custom_model",
        )
        version = result.get("version", "1") if isinstance(result, dict) else "1"
        ui.success(f"Model '{model_name}' registered (version: {version})")
        return f"azureml:{model_name}:{version}"
    except AzureCliError as e:
        ui.error(f"Failed to register model: {e}")
        raise


def create_endpoint(config: PlatoConfig) -> str:
    try:
        run_az(
            "ml", "online-endpoint", "show",
            "--name", config.endpoint_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
        )
        ui.success(f"Endpoint '{config.endpoint_name}' already exists")
    except AzureCliError:
        ui.info(f"Creating endpoint '{config.endpoint_name}'...")
        run_az(
            "ml", "online-endpoint", "create",
            "--name", config.endpoint_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--auth-mode", "key",
            timeout=300,
        )
        ui.success(f"Endpoint '{config.endpoint_name}' created")
    return config.endpoint_name


def create_deployment(
    config: PlatoConfig,
    model_id: str,
    score_script_path: str,
) -> None:
    deployment_name = "default"

    try:
        run_az(
            "ml", "online-deployment", "show",
            "--name", deployment_name,
            "--endpoint-name", config.endpoint_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
        )
        ui.success(f"Deployment '{deployment_name}' already exists")
        return
    except AzureCliError:
        pass

    ui.info(f"Creating deployment '{deployment_name}' (this may take 5-10 minutes)...")

    deployment_yaml = {
        "$schema": "https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json",
        "name": deployment_name,
        "endpoint_name": config.endpoint_name,
        "model": model_id,
        "code_configuration": {
            "code": str(Path(score_script_path).parent),
            "scoring_script": Path(score_script_path).name,
        },
        "environment": f"azureml:{config.environment_name}@latest",
        "instance_type": "Standard_DS2_v2",
        "instance_count": 1,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(deployment_yaml, f)
        deployment_file = f.name

    try:
        run_az(
            "ml", "online-deployment", "create",
            "--file", deployment_file,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
            "--all-traffic",
            timeout=600,
        )
        ui.success(f"Deployment '{deployment_name}' created and serving traffic")
    except AzureCliError as e:
        ui.error(f"Failed to create deployment: {e}")
        raise


def test_endpoint(config: PlatoConfig) -> None:
    test_data = {"state": {"value": 5}}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(test_data, f)
        test_file = f.name

    ui.info("Testing endpoint with sample request...")
    try:
        result = run_az(
            "ml", "online-endpoint", "invoke",
            "--name", config.endpoint_name,
            "--request-file", test_file,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
        )
        ui.success(f"Endpoint response: {result}")
    except AzureCliError as e:
        ui.warn(f"Endpoint test failed (may need warm-up time): {e}")


def get_endpoint_info(config: PlatoConfig) -> None:
    try:
        result = run_az(
            "ml", "online-endpoint", "show",
            "--name", config.endpoint_name,
            "--resource-group", config.resource_group,
            "--workspace-name", config.workspace_name,
        )
        if isinstance(result, dict):
            scoring_uri = result.get("scoring_uri", "N/A")
            ui.info(f"Endpoint URL: {scoring_uri}")
            ui.info("Get auth key with:")
            ui.info(
                f"  az ml online-endpoint get-credentials "
                f"--name {config.endpoint_name} "
                f"--resource-group {config.resource_group} "
                f"--workspace-name {config.workspace_name}"
            )
    except AzureCliError:
        pass


def deploy_agent(config: PlatoConfig, checkpoint_path: str) -> None:
    deploy_src = Path(config.project_root) / "aml_src" / "deploy"
    score_script = str(deploy_src / "score.py")
    model_id = register_model(config, checkpoint_path)
    create_endpoint(config)
    create_deployment(config, model_id, score_script)
    test_endpoint(config)
    get_endpoint_info(config)
    ui.cost_warning()
