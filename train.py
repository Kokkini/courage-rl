import os
import ray
from ray import tune
import argparse
import yaml

from ppo import CustomPPOTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms

args = argparse.ArgumentParser()
args.add_argument("--baseline", action="store_true", help="whether to use the baseline ppo method")
args.add_argument("--config", help="the config file path")
args.add_argument("--tune-config", help="config for hyperparamter tuning")

args = args.parse_args()

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

if args.tune_config is None:
    tune.run(trainer, config=config)
else:
    from hyperopt import hp
    from ray.tune.suggest.hyperopt import HyperOptSearch
    from hyperopt.pyll.base import scope
    import importlib

    tuning_module = importlib.import_module(f"tuning.{args.tune_config}")
    algo = tuning_module.algo

    tune.run(trainer, config=config, search_alg=algo)
