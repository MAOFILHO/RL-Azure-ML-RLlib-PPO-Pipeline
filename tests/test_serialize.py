import json

import gymnasium.spaces as spaces
import numpy as np

from plato_rl.platotk.serialize import GymEncoder, check_and_transform


def test_gym_encoder_scalar():
    val = np.float32(3.14)
    result = json.dumps(val, cls=GymEncoder)
    assert isinstance(json.loads(result), float)


def test_gym_encoder_array_singleton():
    val = np.array([42.0])
    result = json.dumps(val, cls=GymEncoder)
    assert json.loads(result) == 42.0


def test_gym_encoder_array():
    val = np.array([1.0, 2.0, 3.0])
    result = json.dumps(val, cls=GymEncoder)
    assert json.loads(result) == [1.0, 2.0, 3.0]


def test_check_and_transform_box():
    space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    state = [0.5, -0.5]
    result = check_and_transform(space, state)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result.dtype == np.float32


def test_check_and_transform_discrete():
    space = spaces.Discrete(5)
    result = check_and_transform(space, 3.0)
    assert result == 3
    assert isinstance(result, int)


def test_check_and_transform_dict():
    space = spaces.Dict({
        "value": spaces.Box(low=-float("inf"), high=float("inf")),
    })
    state = {"value": 5}
    result = check_and_transform(space, state)
    assert isinstance(result, dict)
    assert "value" in result
    assert isinstance(result["value"], np.ndarray)
