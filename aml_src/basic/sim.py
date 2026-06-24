"""SimpleAdder simulation environment for AML training."""
from typing import Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict


class SimpleAdder(Env):
    def __init__(self, env_config=None):
        self.observation_space = Dict(
            {"value": Box(low=-float("inf"), high=float("inf"))}
        )
        self.action_space = Dict({"addend": Box(low=-10, high=10, dtype=np.int32)})
        self.state = {"value": 0}
        self.iter = 0

    def _get_obs(self):
        return {"value": np.array([self.state["value"]])}

    def _get_info(self):
        return {}

    def reward(self, state):
        return -abs(state["value"] - 50)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.iter = 0
        if options is not None and options.get("value") is not None:
            self.state = {"value": options["value"]}
        else:
            self.state = {"value": np.random.randint(0, 100)}
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.state["value"] += action["addend"].item()
        self.iter += 1
        reward = self.reward(self.state)
        terminated = self.state["value"] == 50
        truncated = self.iter >= 10
        return self._get_obs(), reward, terminated, truncated, self._get_info()
