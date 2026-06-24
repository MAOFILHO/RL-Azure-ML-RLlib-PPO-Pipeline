import json

import gymnasium.spaces as spaces
import numpy as np


class GymEncoder(json.JSONEncoder):
    """Encode a Gymnasium-like state to JSON.

    Encodes an array of shape (1,) into a scalar because Gymnasium only
    supports scalars when using spaces.Discrete.
    """

    def default(self, obj):
        if isinstance(obj, np.number) or isinstance(obj, np.bool_):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            if obj.shape == (1,):
                return obj.item()
            else:
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_and_transform(observation_space, state):
    """Check and transform an observation state for a given observation space."""
    if isinstance(state, np.ndarray) and observation_space.contains(state):
        return state
    elif isinstance(observation_space, spaces.Box):
        return np.array(state, dtype=observation_space.dtype).reshape(
            observation_space.shape
        )
    elif isinstance(observation_space, spaces.Discrete):
        return int(state)
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: check_and_transform(observation_space.spaces[key], state[key])
            for key in observation_space.spaces
        }
    else:
        raise ValueError(f"Cannot transform {state} for {observation_space}")
