from plato_rl.config import PlatoConfig, load_config


def test_config_defaults():
    config = PlatoConfig()
    assert config.resource_group == "plato-rl-rg"
    assert config.location == "eastus"
    assert config.compute_min_nodes == 0
    assert config.compute_max_nodes == 2


def test_config_validate_missing_subscription():
    config = PlatoConfig(subscription_id="")
    errors = config.validate()
    assert len(errors) == 1
    assert "AZURE_SUBSCRIPTION_ID" in errors[0]


def test_config_validate_ok():
    config = PlatoConfig(subscription_id="00000000-0000-0000-0000-000000000000")
    errors = config.validate()
    assert len(errors) == 0


def test_load_config_from_env(env_config):
    config = load_config(env_config)
    assert config.subscription_id == "00000000-0000-0000-0000-000000000000"
    assert config.resource_group == "test-rg"
    assert config.workspace_name == "test-ws"
