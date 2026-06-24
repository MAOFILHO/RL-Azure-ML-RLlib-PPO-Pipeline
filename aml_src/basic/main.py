"""Training script for SimpleAdder on AML with trajectory logging and MLflow."""
import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import mlflow
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from sim import SimpleAdder as SimEnv

register_env("sim_env", lambda config: SimEnv(config))


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
            out_dir = Path("./outputs/trajectories")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{dt.datetime.now().isoformat(timespec='milliseconds')}.csv"
            self.fname = out_dir / output_name
            print(f"[plato] Saving trajectories to {self.fname}")

        header = ("episode_id", "step_id", "state", "action", "reward", "terminated", "truncated")
        first = not Path(self.fname).exists()

        with open(self.fname, "a") as fp:
            writer = csv.writer(fp)
            if first:
                writer.writerow(header)
            for row in zip(episode_id, step_id, obs, actions, rewards, terminated, truncated):
                writer.writerow(row)


def train(local=False, iterations=10):
    print(f"[plato] Starting PPO training for {iterations} iterations")

    if not local:
        mlflow.start_run()
        mlflow.log_params({"iterations": iterations, "train_batch_size": 4000, "algorithm": "PPO"})

    algo = (
        PPOConfig()
        .callbacks(TrajectoryCallback)
        .rollouts(num_rollout_workers=1 if not local else 0)
        .resources(num_gpus=0)
        .training(train_batch_size=4_000)
        .environment(env="sim_env")
        .build()
    )

    for i in range(iterations):
        result = algo.train()
        reward_mean = result["episode_reward_mean"]
        print(f"[plato] Iteration {i + 1}/{iterations} — reward_mean: {reward_mean:.2f}")
        if not local:
            mlflow.log_metrics({
                "episode_reward_mean": reward_mean,
                "episode_reward_max": result.get("episode_reward_max", 0),
                "episode_reward_min": result.get("episode_reward_min", 0),
                "episodes_total": result.get("episodes_total", 0),
                "timesteps_total": result.get("timesteps_total", 0),
            }, step=i + 1)

    checkpoint_dir = algo.save(checkpoint_dir="./outputs")
    print(f"[plato] Checkpoint saved: {checkpoint_dir}")
    if not local:
        mlflow.log_artifact(checkpoint_dir)
        mlflow.end_run()
    algo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=False)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if args.test_local:
        train(local=True, iterations=args.iterations)
        sys.exit()

    from ray_on_aml.core import Ray_On_AML
    ray_on_aml = Ray_On_AML()
    ray = ray_on_aml.getRay()
    if ray:
        print("[plato] Head node detected")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        train(iterations=args.iterations)
    else:
        print("[plato] Worker node")
