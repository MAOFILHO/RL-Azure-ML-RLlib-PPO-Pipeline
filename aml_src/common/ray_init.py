"""Shared Ray initialization for AML jobs."""
import sys


def init_ray_on_aml():
    """Initialize Ray on AML. Returns True on head node, False on worker."""
    from ray_on_aml.core import Ray_On_AML

    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()

    if ray:
        print("[plato] Head node detected — initializing Ray cluster")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        return True
    else:
        print("[plato] Worker node — waiting for head node")
        return False


def init_ray_local():
    """Initialize Ray for local testing."""
    print("[plato] Running in local mode")
