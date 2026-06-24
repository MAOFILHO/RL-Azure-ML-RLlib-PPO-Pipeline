"""Curriculum-capable SimpleAdder for AML training."""
import numpy as np
from gymnasium.spaces import Box, Dict
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.utils.annotations import override


class SimpleAdder(TaskSettableEnv):
    def __init__(self, env_config=None):
        self.observation_space = Dict(
            {"value": Box(low=-float("inf"), high=float("inf"))}
        )
        self.action_space = Dict({"addend": Box(low=-10, high=10, dtype=np.int32)})
        self.exponent = 1
        self.iter = 0
        self.state = {"value": 0}

    def _get_obs(self):
        return {"value": np.array([self.state["value"]])}

    def _get_info(self):
        return {}

    def reward(self, state):
        return -abs(state["value"] - 50) + 10 - self.iter

    def reset(self, *, seed=None, options=None):
        self.iter = 0
        task = self.get_task()
        exponent = task["exponent"]
        self.state = {"value": 50 + np.random.randint(-(2**exponent), 2**exponent)}
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.state["value"] += action["addend"].item()
        self.iter += 1
        reward = self.reward(self.state)
        terminated = self.state["value"] == 50
        truncated = self.iter >= 10
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    @override(TaskSettableEnv)
    def get_task(self):
        return {"exponent": self.exponent}

    @override(TaskSettableEnv)
    def set_task(self, task):
        self.exponent = task["exponent"]
