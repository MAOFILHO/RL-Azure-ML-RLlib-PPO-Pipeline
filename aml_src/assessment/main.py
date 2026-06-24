"""Custom assessment script for AML — evaluates trained agent on specific configs."""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import ray
from ray.tune.registry import register_env

from sim import SimpleAdder as SimEnv

register_env("sim_env", lambda conf: SimEnv(conf))


@ray.remote
class EpisodeRunner:
    def __init__(self, checkpoint_folder: str):
        checkpoint_path = Path(checkpoint_folder)
        with open(checkpoint_path / "algorithm_state.pkl", "rb") as fp:
            data = pickle.load(fp)
        agent_config = data["config"].copy(copy_frozen=False)
        self.agent = agent_config.rollouts(num_rollout_workers=0).build()
        self.agent.restore(checkpoint_path)
        self.sim = SimEnv(env_config={})

    def run_episode(
        self, initial_condition: Dict[str, int], episode_id: int = 0, max_steps: int = 100
    ) -> pd.DataFrame:
        self.initial_condition = initial_condition
        observable_state, _ = self.sim.reset(options=initial_condition)

        states, actions, rewards, terminated, truncated = [], [], [], [], []
        step = 0
        for step in range(max_steps):
            full_state = copy.deepcopy(self.sim.state)
            states.append(full_state)
            action = self.agent.compute_single_action(observable_state, explore=False)
            observable_state, reward, terminate, truncate, _ = self.sim.step(action)
            rewards.append(reward)
            actions.append(action)
            terminated.append(terminate)
            truncated.append(truncate)
            if terminate or truncate:
                break

        states.append(
            self.sim.state.copy() if isinstance(self.sim.state, dict) else self.sim.state
        )

        df = pd.DataFrame(states).add_prefix("state_")
        df = pd.concat([df, pd.DataFrame(actions).add_prefix("action_")], axis=1)
        if isinstance(initial_condition, dict):
            df[[f"config_{k}" for k in initial_condition.keys()]] = list(
                initial_condition.values()
            )
        df["reward"] = pd.Series(rewards)
        df["reward_cumsum"] = df["reward"].cumsum()
        df["terminated"] = pd.Series(terminated)
        df["truncated"] = pd.Series(truncated)
        df["episode_id"] = episode_id
        df["step_id"] = pd.Series(np.arange(step + 2, dtype=int))
        return df


def run_episodes(checkpoint_folder, initial_conditions_json):
    initial_conditions = json.load(open(initial_conditions_json, "r"))
    num_workers = min(4, len(initial_conditions))
    actors = [EpisodeRunner.remote(checkpoint_folder) for _ in range(num_workers)]

    results = []
    for idx, init_cond in enumerate(initial_conditions):
        actor = actors[idx % num_workers]
        result_id = actor.run_episode.remote(init_cond, idx)
        results.append(result_id)

    results_df = pd.DataFrame()
    for result_id in results:
        episode_df = ray.get(result_id)
        results_df = pd.concat([results_df, episode_df], axis=0)

    outdir = "./outputs"
    os.makedirs(outdir, exist_ok=True)
    results_df.to_csv(f"{outdir}/assessment_logs.csv", mode="w", header=True, index=False)
    print(f"[plato] Assessment results saved to {outdir}/assessment_logs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-local", action="store_true", default=False)
    parser.add_argument("--checkpoint-folder", default="checkpoints/checkpoint_000010")
    parser.add_argument("--input-json", default="init_conditions.json")
    args = parser.parse_args()

    if args.test_local:
        run_episodes(args.checkpoint_folder, args.input_json)
        sys.exit()

    from ray_on_aml.core import Ray_On_AML
    ray_on_aml = Ray_On_AML()
    ray_instance = ray_on_aml.getRay()
    if ray_instance:
        print("[plato] Head node detected")
        ray.init(address="auto")
        print(f"[plato] Cluster resources: {ray.cluster_resources()}")
        run_episodes(args.checkpoint_folder, args.input_json)
    else:
        print("[plato] Worker node")
