from unittest.mock import patch

from plato_rl.azure.teardown import list_resources, check_orphaned_resources
from plato_rl.config import PlatoConfig


@patch("plato_rl.azure.teardown.run_az")
def test_list_resources(mock_run):
    mock_run.return_value = [
        {"name": "ws", "type": "Microsoft.MachineLearningServices/workspaces", "location": "eastus"},
        {"name": "compute", "type": "Microsoft.MachineLearningServices/workspaces/computes", "location": "eastus"},
    ]
    config = PlatoConfig(resource_group="test-rg")
    resources = list_resources(config)
    assert len(resources) == 2


@patch("plato_rl.azure.teardown.run_az")
def test_check_no_orphans(mock_run):
    mock_run.return_value = []
    orphans = check_orphaned_resources()
    assert len(orphans) == 0
