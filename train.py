from ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms
load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ray.init()
config={
  "env": dangerous_maze_env.DangerousMazeEnv,
  "num_workers": 2,
  "num_cpus_per_worker": 0.5,
  "model": {"custom_model": "vision_net",
            "custom_options": {}}
}

tune.run(PPOTrainer, config=config)