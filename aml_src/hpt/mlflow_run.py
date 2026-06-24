"""Hyperparameter tuning with Ray Tune and MLflow on AML."""
import argparse
import sys

import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler


def run(
    num_tune_samples: int = 4,
    env_name: str = "CartPole-v1",
) -> tune.ResultGrid:
    tune_sched = ASHAScheduler(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        max_t=10,
        grace_period=3,
        reduction_factor=3,
        brackets=1,
    )

    stopping_criteria = {"training_iteration": 10}

    current_run = mlflow.active_run()
    if current_run is None:
        current_run = mlflow.start_run()

    print(f"[plato] Starting HPT with {num_tune_samples} samples on {env_name}")
    print(f"[plato] MLflow run ID: {current_run.info.run_id}")

    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(
            scheduler=tune_sched,
            num_samples=num_tune_samples,
        ),
        param_space={
            "env": env_name,
            "kl_coeff": 1.0,
            "lambda": tune.choice([0.9, 0.95, 1.0]),
            "clip_param": tune.choice([0.1, 0.2, 0.3]),
            "lr": tune.choice([1e-3, 5e-4, 1e-4]),
            "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": 512,
            "train_batch_size": 2000,
            "num_workers": 1,
            "num_gpus": 0,
            "framework": "torch",
        },
        run_config=air.RunConfig(
            stop=stopping_criteria,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
            ),
            callbacks=[
                MLflowLoggerCallback(
                    tags={MLFLOW_PARENT_RUN_ID: current_run.info.run_id},
                    experiment_name="plato_hpt_ppo",
                    save_artifact=True,
                )
            ],
            verbose=2,
        ),
    )
    results = tuner.fit()

    best = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"[plato] Best trial config: {best.config}")
    print(f"[plato] Best trial reward: {best.metrics.get('episode_reward_mean', 'N/A')}")
    print("[plato] Hyperparameter tuning completed")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true")
    parser.add_argument("--num-tune-samples", type=int, default=4)
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    args, _ = parser.parse_known_args()

    if args.test_local:
        run(args.num_tune_samples, args.env_name)
        sys.exit()

    from ray_on_aml.core import Ray_On_AML
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:
        print("[plato] Head node detected")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        run(args.num_tune_samples, args.env_name)
    else:
        print("[plato] Worker node")
