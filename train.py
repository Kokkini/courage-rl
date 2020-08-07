import os
import ray
from ray import tune
import argparse
import yaml
import importlib

from ppo import StateDangerPPOTrainer, ActionDangerPPOTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms

args = argparse.ArgumentParser()
args.add_argument("--baseline", action="store_true", help="whether to use the baseline ppo method")
args.add_argument("--config", help="the config file path")
args.add_argument("--tune-config", help="config for hyperparamter tuning")
args.add_argument("--num-tune-runs", type=int, default=20, help="number of hyperparamter sets to test")
args.add_argument("--action-danger", action="store_true", help="whether to calculate danger level using action instead of state")

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
    config["model"]["custom_options"]["action_danger"] = args.action_danger
    if args.action_danger:
        trainer = ActionDangerPPOTrainer
    else:
        trainer = StateDangerPPOTrainer

stop = None
if "stop" in config:
    stop = config.pop("stop")

if args.tune_config is None:
    tune.run(trainer, config=config, stop=stop)
else:
    tuning_module = importlib.import_module(f"tuning.{args.tune_config}")
    algo = tuning_module.algo
    print(config)
    tune.run(trainer, config=config, search_alg=algo, stop=stop, num_samples=args.num_tune_runs)
