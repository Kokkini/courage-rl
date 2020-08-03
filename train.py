import os
import ray
from ray import tune
import argparse

import dangerous_maze_env
from ppo import CustomPPOTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms

args = argparse.ArgumentParser()
args.add_argument("--baseline", action="store_true", help="whether to use the baseline ppo method")
args.add_argument("--config", help="the config file path")
args.parse_args()

load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ray.init()
with open(args.config) as f:
    config = yaml.safe_load(f)

trainer = None
if args.baseline:
    trainer = PPOTrainer
else:
    trainer = CustomPPOTrainer

tune.run(trainer, config=config)
