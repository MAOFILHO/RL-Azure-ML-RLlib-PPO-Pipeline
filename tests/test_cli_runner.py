import json
from unittest.mock import patch, MagicMock

import pytest

from plato_rl.azure.cli_runner import (
    AzureCliError,
    run_az,
    check_az_installed,
    check_az_ml_extension,
    get_current_subscription,
)


def _mock_run(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


@patch("subprocess.run")
def test_run_az_success(mock_run):
    mock_run.return_value = _mock_run(stdout='{"name": "test-rg"}')
    result = run_az("group", "show", "--name", "test-rg")
    assert result == {"name": "test-rg"}


@patch("subprocess.run")
def test_run_az_failure(mock_run):
    mock_run.return_value = _mock_run(returncode=1, stderr="Resource not found")
    with pytest.raises(AzureCliError) as exc_info:
        run_az("group", "show", "--name", "nonexistent")
    assert "Resource not found" in str(exc_info.value)


@patch("subprocess.run")
def test_run_az_retry_on_429(mock_run):
    fail = _mock_run(returncode=1, stderr="429 Too Many Requests")
    success = _mock_run(stdout='{"ok": true}')
    mock_run.side_effect = [fail, success]
    result = run_az("group", "show", "--name", "test", retry=2, backoff=0.01)
    assert result == {"ok": True}
    assert mock_run.call_count == 2


@patch("subprocess.run")
def test_check_az_installed_true(mock_run):
    mock_run.return_value = _mock_run(stdout="{}")
    assert check_az_installed() is True


@patch("subprocess.run")
def test_check_az_installed_false(mock_run):
    mock_run.side_effect = FileNotFoundError()
    assert check_az_installed() is False


@patch("subprocess.run")
def test_check_az_ml_extension(mock_run):
    mock_run.return_value = _mock_run(
        stdout=json.dumps([{"name": "ml"}, {"name": "other"}])
    )
    assert check_az_ml_extension() is True


@patch("subprocess.run")
def test_check_az_ml_extension_missing(mock_run):
    mock_run.return_value = _mock_run(stdout=json.dumps([{"name": "other"}]))
    assert check_az_ml_extension() is False


@patch("plato_rl.azure.cli_runner.run_az")
def test_get_current_subscription(mock_run_az):
    mock_run_az.return_value = {"id": "sub-123", "name": "My Sub"}
    assert get_current_subscription() == "sub-123"
