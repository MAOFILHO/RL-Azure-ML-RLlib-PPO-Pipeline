import json
import subprocess
import time
from typing import Any, Optional


class AzureCliError(Exception):
    def __init__(self, command: str, returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"az command failed (rc={returncode}): {stderr.strip()}")


def run_az(
    *args: str,
    retry: int = 3,
    backoff: float = 2.0,
    parse_json: bool = True,
    timeout: int = 600,
) -> Optional[Any]:
    cmd = ["az"] + list(args)
    if parse_json:
        cmd += ["--output", "json"]

    last_error = None
    for attempt in range(retry):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            last_error = AzureCliError(" ".join(cmd), -1, "Command timed out")
            time.sleep(backoff * (2**attempt))
            continue

        if result.returncode == 0:
            if parse_json and result.stdout.strip():
                return json.loads(result.stdout)
            return result.stdout.strip()

        stderr = result.stderr
        if "429" in stderr or "503" in stderr or "Retry" in stderr:
            last_error = AzureCliError(" ".join(cmd), result.returncode, stderr)
            time.sleep(backoff * (2**attempt))
            continue

        raise AzureCliError(" ".join(cmd), result.returncode, stderr)

    raise last_error  # type: ignore[misc]


def check_az_installed() -> bool:
    try:
        result = subprocess.run(
            ["az", "version"], capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_az_ml_extension() -> bool:
    try:
        result = subprocess.run(
            ["az", "extension", "list", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return False
        extensions = json.loads(result.stdout)
        return any(ext.get("name") == "ml" for ext in extensions)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return False


def get_current_subscription() -> Optional[str]:
    try:
        result = run_az("account", "show")
        if isinstance(result, dict):
            return result.get("id")
    except AzureCliError:
        pass
    return None
