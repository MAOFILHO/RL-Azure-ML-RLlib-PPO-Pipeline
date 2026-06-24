"""Curriculum learning training script for AML with MLflow."""
import argparse
import sys

import mlflow
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

from sim_curriculum_capable import SimpleAdder as CurriculumCapableEnv

register_env("curriculum_env", lambda config: CurriculumCapableEnv(config))


def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    reward_threshold = 0
    task_exponent = task_settable_env.get_task()["exponent"]
    avg_reward = train_results["episode_reward_mean"]
    if avg_reward >= reward_threshold:
        return {"exponent": task_exponent + 1}
    else:
        return task_settable_env.get_task()


class CurriculumCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        task = base_env.get_sub_environments()[env_index].get_task()
        episode.custom_metrics["task"] = task["exponent"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        print(f"[plato] Iteration: {result['training_iteration']} — "
              f"reward_mean: {result['episode_reward_mean']:.2f}, "
              f"task: {result['custom_metrics'].get('task_mean', 'N/A')}")
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)


def train(iterations=300, max_timesteps=100000, local=False):
    print(f"[plato] Starting curriculum learning (max {iterations} iterations, {max_timesteps} timesteps)")

    if not local:
        mlflow.start_run()
        mlflow.log_params({"iterations": iterations, "max_timesteps": max_timesteps, "algorithm": "PPO", "mode": "curriculum"})

    algo = (
        PPOConfig()
        .callbacks(CurriculumCallback)
        .rollouts(num_rollout_workers=1 if not local else 0)
        .resources(num_gpus=0)
        .training(train_batch_size=4_000)
        .environment(env="curriculum_env", env_task_fn=curriculum_fn)
        .build()
    )

    total_timesteps = 0
    for i in range(iterations):
        result = algo.train()
        total_timesteps = result.get("timesteps_total", 0)
        reward_mean = result["episode_reward_mean"]
        task_level = result.get("custom_metrics", {}).get("task_mean", 0)
        print(f"[plato] Iteration {i + 1}/{iterations} — "
              f"reward_mean: {reward_mean:.2f}, "
              f"timesteps: {total_timesteps}")
        if not local:
            mlflow.log_metrics({
                "episode_reward_mean": reward_mean,
                "episode_reward_max": result.get("episode_reward_max", 0),
                "timesteps_total": total_timesteps,
                "task_level": task_level,
            }, step=i + 1)
        if total_timesteps >= max_timesteps:
            print(f"[plato] Reached {max_timesteps} timesteps, stopping")
            break

    checkpoint_dir = algo.save(checkpoint_dir="./outputs")
    print(f"[plato] Checkpoint saved: {checkpoint_dir}")
    if not local:
        mlflow.log_artifact(checkpoint_dir)
        mlflow.end_run()
    algo.stop()
    print("[plato] Curriculum learning completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--max-timesteps", type=int, default=100000)
    args = parser.parse_args()

    if args.test_local:
        train(iterations=args.iterations, max_timesteps=args.max_timesteps, local=True)
        sys.exit()

    from ray_on_aml.core import Ray_On_AML
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:
        print("[plato] Head node detected")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        train(iterations=args.iterations, max_timesteps=args.max_timesteps)
        ray.shutdown()
    else:
        print("[plato] Worker node")
