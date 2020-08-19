import os
import ray
from ray import tune
import argparse
import yaml
import importlib


from callbacks import CustomCallbacks
from ppo import StateDangerPPOTrainer, ActionDangerPPOTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms

args = argparse.ArgumentParser()
args.add_argument("--baseline", action="store_true", help="whether to use the baseline ppo method")
args.add_argument("--config", help="the config file path")
args.add_argument("--tune-config", help="config for hyperparamter tuning")
args.add_argument("--num-tune-runs", type=int, default=20, help="number of hyperparamter sets to test")
args.add_argument("--state-danger", action="store_true", help="whether to calculate danger level for states instead of state-action pairs, default is state-action pairs")
args.add_argument("--callback", action="store_true")
args.add_argument("--visual_obs", action="store_true", help="whether the observation is visual (an image) or non visual (a vector), default is non visual")

args = args.parse_args()

load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ray.init()
with open(args.config) as f:
    config = yaml.safe_load(f)

trainer = None

if args.visual_obs:
    config["model"]["custom_model"] = "vision_net"
else:
    config["model"]["custom_model"] = "simple_fcnet"

if args.baseline:
    trainer = PPOTrainer
else:
    config["model"]["custom_options"]["state_danger"] = args.state_danger
    if not args.state_danger:
        trainer = ActionDangerPPOTrainer
    else:
        trainer = StateDangerPPOTrainer


if args.callback:
    config["callbacks"] = CustomCallbacks

stop = None
if "stop" in config:
    stop = config.pop("stop")

if not args.baseline:
    env = trainer(config=config, env=config["env"]).env_creator(config.get("env_config"))
    env_max_step = env.spec.max_episode_steps
    env.close()
    config["max_step"] = env_max_step

    if config.get("use_curiosity", False):
        config["model"]["custom_options"]["use_curiosity"] = True

print(config)

if args.tune_config is None:
    tune.run(trainer, config=config, stop=stop)
else:
    tuning_module = importlib.import_module(f"tuning.{args.tune_config}")
    algo = tuning_module.algo
    print(config)
    tune.run(trainer, config=config, search_alg=algo, stop=stop, num_samples=args.num_tune_runs)
