"""AML managed endpoint scoring script."""
import json
import logging
import os
from pathlib import Path

import gymnasium.spaces as spaces
import numpy as np

from platotk.restore import restore_agent_from_pickle
from platotk.serialize import GymEncoder, check_and_transform

observation_space = spaces.Dict(
    {"value": spaces.Box(low=-float("inf"), high=float("inf"))}
)
action_space = spaces.Dict({"addend": spaces.Box(low=-10, high=10, dtype=np.int32)})


def init():
    global model
    model_dir = Path(os.getenv("AZUREML_MODEL_DIR"))
    checkpoint_folder = model_dir
    if (model_dir / "checkpoint_000010").exists():
        checkpoint_folder = model_dir / "checkpoint_000010"
    logging.info(f"Loading agent from {checkpoint_folder}")
    model = restore_agent_from_pickle(
        observation_space, action_space, checkpoint_folder
    )
    logging.info("Agent initialized from checkpoint")


def run(raw_state):
    logging.info("Request received")
    state = json.loads(raw_state)["state"]
    state = check_and_transform(observation_space, state)
    action = model.compute_single_action(state, explore=False)
    logging.info("Action computed")
    return json.loads(json.dumps(action, cls=GymEncoder))
