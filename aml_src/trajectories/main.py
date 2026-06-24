"""Trajectory logging training script for AML with MLflow."""
import argparse
import csv
import datetime as dt
from pathlib import Path

import mlflow
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig


parser = argparse.ArgumentParser()
parser.add_argument("--storage-path", type=Path, default=Path("./outputs/trajectories"))
args, _ = parser.parse_known_args()


class TrajectoryCallback(DefaultCallbacks):
    fname = None

    def on_postprocess_trajectory(
        self, *, worker, episode, agent_id, policy_id, policies,
        postprocessed_batch, original_batches,
    ):
        obs = postprocessed_batch["obs"]
        actions = postprocessed_batch["actions"]
        episode_id = postprocessed_batch["eps_id"]
        rewards = postprocessed_batch["rewards"]
        terminated = postprocessed_batch["terminateds"]
        truncated = postprocessed_batch["truncateds"]
        step_id = postprocessed_batch["t"]

        if self.fname is None:
            args.storage_path.mkdir(parents=True, exist_ok=True)
            output_name = f"{dt.datetime.now().isoformat(timespec='milliseconds')}.csv"
            self.fname = args.storage_path / output_name
            print(f"[plato] Saving trajectories to {self.fname}")

        header = ("episode_id", "step_id", "state", "action", "reward", "terminated", "truncated")
        first = not Path(self.fname).exists()

        if not first:
            with open(self.fname, "r") as fp:
                reader = csv.reader(fp)
                file_head = tuple(next(reader))
                if file_head != header:
                    raise ValueError(f"Unexpected header in file {self.fname}")

        with open(self.fname, "a") as fp:
            writer = csv.writer(fp)
            if first:
                writer.writerow(header)
            for row in zip(episode_id, step_id, obs, actions, rewards, terminated, truncated):
                writer.writerow(row)


def train():
    print("[plato] Training CartPole-v1 with trajectory logging + MLflow")
    mlflow.start_run()
    mlflow.log_params({"env": "CartPole-v1", "iterations": 10, "algorithm": "PPO", "mode": "trajectory_logging"})

    algo = (
        PPOConfig()
        .callbacks(TrajectoryCallback)
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .training(train_batch_size=4_000)
        .environment(env="CartPole-v1")
        .build()
    )

    for i in range(10):
        result = algo.train()
        reward_mean = result["episode_reward_mean"]
        print(f"[plato] Iteration {i + 1}/10 — reward_mean: {reward_mean:.2f}")
        mlflow.log_metrics({
            "episode_reward_mean": reward_mean,
            "episode_reward_max": result.get("episode_reward_max", 0),
            "episodes_total": result.get("episodes_total", 0),
            "timesteps_total": result.get("timesteps_total", 0),
        }, step=i + 1)

    checkpoint_dir = algo.save(checkpoint_dir="./outputs")
    print(f"[plato] Checkpoint saved: {checkpoint_dir}")
    if TrajectoryCallback.fname:
        mlflow.log_artifact(str(TrajectoryCallback.fname))
    mlflow.end_run()
    algo.stop()


if __name__ == "__main__":
    from ray_on_aml.core import Ray_On_AML
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:
        print("[plato] Head node detected")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        train()
    else:
        print("[plato] Worker node")
