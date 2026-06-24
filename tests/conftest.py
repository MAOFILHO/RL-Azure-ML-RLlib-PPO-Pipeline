import os
import pytest


@pytest.fixture
def env_config(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AZURE_SUBSCRIPTION_ID=00000000-0000-0000-0000-000000000000\n"
        "AZURE_RESOURCE_GROUP=test-rg\n"
        "AZURE_LOCATION=eastus\n"
        "AZURE_WORKSPACE_NAME=test-ws\n"
    )
    return env_file
