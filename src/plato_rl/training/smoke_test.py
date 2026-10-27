"""Local smoke test — validates the full RL stack without Azure."""
from plato_rl import console as ui


def run_smoke_test(iterations: int = 5, batch_size: int = 2000) -> bool:
    ui.info("Importing Ray RLlib (this may take a moment on first run)...")

    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    from plato_rl.sims.simple_adder import SimpleAdder

    register_env("smoke_test_env", lambda config: SimpleAdder(config))

    ui.info(f"Training PPO on SimpleAdder for {iterations} iterations (local)...")

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=0)
        .resources(num_gpus=0)
        .training(train_batch_size=batch_size)
        .environment(env="smoke_test_env")
        .build()
    )

    rewards = []
    for i in range(iterations):
        result = algo.train()
        reward = result["episode_reward_mean"]
        rewards.append(reward)
        ui.info(
            f"  Iteration {i + 1}/{iterations} — "
            f"mean reward: {reward:.2f}, "
            f"episodes: {result['episodes_this_iter']}"
        )

    checkpoint_dir = algo.save(checkpoint_dir="./outputs/smoke_test")
    ui.success(f"Checkpoint saved: {checkpoint_dir}")

    algo.stop()

    if len(rewards) >= 2 and rewards[-1] > rewards[0]:
        ui.success("Agent is learning — reward improved during training")
    else:
        ui.warn("Reward did not clearly improve — this is OK for a short smoke test")

    return True
