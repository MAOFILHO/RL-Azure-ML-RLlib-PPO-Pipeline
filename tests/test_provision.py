from unittest.mock import patch, call

from plato_rl.azure.cli_runner import AzureCliError
from plato_rl.azure.provision import ensure_resource_group, ensure_workspace
from plato_rl.config import PlatoConfig


@patch("plato_rl.azure.provision.run_az")
def test_ensure_resource_group_exists(mock_run):
    mock_run.return_value = {"name": "test-rg"}
    config = PlatoConfig(resource_group="test-rg", location="eastus")
    ensure_resource_group(config)
    mock_run.assert_called_once()


@patch("plato_rl.azure.provision.run_az")
def test_ensure_resource_group_creates(mock_run):
    mock_run.side_effect = [
        AzureCliError("show", 1, "Not found"),
        {"name": "test-rg"},
    ]
    config = PlatoConfig(resource_group="test-rg", location="eastus")
    ensure_resource_group(config)
    assert mock_run.call_count == 2


@patch("plato_rl.azure.provision.run_az")
def test_ensure_workspace_exists(mock_run):
    mock_run.return_value = {"name": "test-ws"}
    config = PlatoConfig(
        resource_group="test-rg",
        workspace_name="test-ws",
        location="eastus",
    )
    ensure_workspace(config)
    mock_run.assert_called_once()
