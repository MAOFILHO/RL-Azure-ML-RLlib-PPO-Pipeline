from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


@dataclass
class PlatoConfig:
    subscription_id: str = ""
    resource_group: str = "plato-rl-rg"
    location: str = "eastus"
    workspace_name: str = "plato-rl-ws"
    compute_name: str = "plato-compute"
    compute_vm_size: str = "Standard_D4s_v3"
    compute_min_nodes: int = 0
    compute_max_nodes: int = 2
    environment_name: str = "plato-rl-env"
    training_iterations: int = 10
    train_batch_size: int = 4000
    num_tune_samples: int = 10
    endpoint_name: str = "plato-agent-endpoint"
    project_root: Path = field(default_factory=lambda: Path.cwd())

    def validate(self) -> list[str]:
        errors = []
        if not self.subscription_id:
            errors.append("AZURE_SUBSCRIPTION_ID is required in .env")
        if not self.resource_group:
            errors.append("AZURE_RESOURCE_GROUP is required")
        if not self.workspace_name:
            errors.append("AZURE_WORKSPACE_NAME is required")
        return errors


def load_config(env_path: Optional[Path] = None) -> PlatoConfig:
    if env_path is None:
        env_path = Path.cwd() / ".env"
    load_dotenv(env_path)

    return PlatoConfig(
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
        resource_group=os.getenv("AZURE_RESOURCE_GROUP", "plato-rl-rg"),
        location=os.getenv("AZURE_LOCATION", "eastus"),
        workspace_name=os.getenv("AZURE_WORKSPACE_NAME", "plato-rl-ws"),
        compute_name=os.getenv("AZURE_COMPUTE_NAME", "plato-compute"),
        compute_vm_size=os.getenv("AZURE_COMPUTE_VM_SIZE", "Standard_D4s_v3"),
        compute_min_nodes=int(os.getenv("AZURE_COMPUTE_MIN_NODES", "0")),
        compute_max_nodes=int(os.getenv("AZURE_COMPUTE_MAX_NODES", "2")),
        environment_name=os.getenv("AZURE_ENVIRONMENT_NAME", "plato-rl-env"),
        training_iterations=int(os.getenv("TRAINING_ITERATIONS", "10")),
        train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "4000")),
        num_tune_samples=int(os.getenv("NUM_TUNE_SAMPLES", "10")),
        endpoint_name=os.getenv("AZURE_ENDPOINT_NAME", "plato-agent-endpoint"),
        project_root=Path.cwd(),
    )
