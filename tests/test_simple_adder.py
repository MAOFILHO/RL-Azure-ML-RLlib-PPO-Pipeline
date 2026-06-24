import numpy as np

from plato_rl.sims.simple_adder import SimpleAdder
from plato_rl.sims.simple_adder_curriculum import SimpleAdderCurriculum


class TestSimpleAdder:
    def test_reset_returns_correct_shape(self):
        env = SimpleAdder()
        obs, info = env.reset()
        assert "value" in obs
        assert obs["value"].shape == (1,)

    def test_step_applies_action(self):
        env = SimpleAdder()
        env.reset(options={"value": 45})
        action = {"addend": np.array([5], dtype=np.int32)}
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["value"].item() == 50
        assert terminated is True
        assert reward == 0

    def test_negative_reward_when_not_at_target(self):
        env = SimpleAdder()
        env.reset(options={"value": 10})
        action = {"addend": np.array([0], dtype=np.int32)}
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward < 0

    def test_truncates_after_10_steps(self):
        env = SimpleAdder()
        env.reset(options={"value": 0})
        for _ in range(10):
            action = {"addend": np.array([0], dtype=np.int32)}
            obs, reward, terminated, truncated, info = env.step(action)
        assert truncated is True

    def test_reset_with_options(self):
        env = SimpleAdder()
        obs, _ = env.reset(options={"value": 42})
        assert obs["value"].item() == 42


class TestSimpleAdderCurriculum:
    def test_get_set_task(self):
        env = SimpleAdderCurriculum()
        assert env.get_task() == {"exponent": 1}
        env.set_task({"exponent": 5})
        assert env.get_task() == {"exponent": 5}

    def test_reset_uses_task(self):
        env = SimpleAdderCurriculum()
        env.set_task({"exponent": 1})
        obs, _ = env.reset()
        value = obs["value"].item()
        assert 48 <= value <= 52

    def test_step_returns_correct_format(self):
        env = SimpleAdderCurriculum()
        env.reset()
        action = {"addend": np.array([1], dtype=np.int32)}
        result = env.step(action)
        assert len(result) == 5
