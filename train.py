import os
import ray
from ray import tune

import dangerous_maze_env
from ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms
load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ray.init()
config={
  "env": dangerous_maze_env.DangerousMazeEnv,
  "num_workers": 2,
  "num_cpus_per_worker": 0.5,
  "num_gpus": 0.5,
  "num_gpus_per_worker": 0.25,
  "num_envs_per_worker": 64,
  "model": {"custom_model": "vision_net",
            "custom_options": {}}
}

tune.run(PPOTrainer, config=config)