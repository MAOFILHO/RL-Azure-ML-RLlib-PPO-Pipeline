"""Reusable callback for logging episode trajectories to CSV."""
import csv
import datetime as dt
from pathlib import Path
from typing import Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class TrajectoryCallback(DefaultCallbacks):
    """Logs episode trajectories (obs, actions, rewards) to a CSV file."""

    def __init__(self, *args, storage_path: Optional[Path] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._storage_path = storage_path
        self._fname: Optional[Path] = None

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
    ):
        obs = postprocessed_batch["obs"]
        actions = postprocessed_batch["actions"]
        episode_id = postprocessed_batch["eps_id"]
        rewards = postprocessed_batch["rewards"]
        terminated = postprocessed_batch["terminateds"]
        truncated = postprocessed_batch["truncateds"]
        step_id = postprocessed_batch["t"]

        if self._fname is None:
            if self._storage_path is None:
                self._storage_path = Path("./outputs/trajectories")
            self._storage_path.mkdir(parents=True, exist_ok=True)
            output_name = f"{dt.datetime.now().isoformat(timespec='milliseconds')}.csv"
            self._fname = self._storage_path / output_name

        header = (
            "episode_id",
            "step_id",
            "state",
            "action",
            "reward",
            "terminated",
            "truncated",
        )

        first = not self._fname.exists()

        if not first:
            with open(self._fname, "r") as fp:
                reader = csv.reader(fp)
                file_head = tuple(next(reader))
                if file_head != header:
                    raise ValueError(f"Unexpected header in file {self._fname}")

        with open(self._fname, "a") as fp:
            writer = csv.writer(fp)
            if first:
                writer.writerow(header)
            for row in zip(
                episode_id, step_id, obs, actions, rewards, terminated, truncated
            ):
                writer.writerow(row)
