import os
import ray
from ray import tune

import dangerous_maze_env
from ppo import CustomPPOTrainer
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
  "num_envs_per_worker": 32,
  "batch_mode": "complete_episodes",
  "max_step": 200,
  "danger_loss_coeff": 1,
  "danger_reward_coeff": 0.1,
  "gamma_danger": 0.9,
  "model": {"custom_model": "vision_net",
            "custom_options": {}}
}

tune.run(CustomPPOTrainer, config=config)