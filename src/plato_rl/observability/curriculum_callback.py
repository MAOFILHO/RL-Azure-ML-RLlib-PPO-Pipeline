"""Callback for logging curriculum learning progress."""
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CurriculumCallback(DefaultCallbacks):
    """Logs curriculum task difficulty to custom metrics."""

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        task = base_env.get_sub_environments()[env_index].get_task()
        episode.custom_metrics["task"] = task["exponent"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
