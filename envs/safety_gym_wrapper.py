import time
import gym
import numpy as np
from ray.tune import registry



class SafetyGymWrapper(gym.Env):
    def __init__(self, config):
        self.config = config
        try:
            import safety_gym
            from safety_gym.envs.engine import Engine
            env = Engine(self.config)
        except Exception as e:
            print(e)
            print("safety gym not installed, using cartpole instead")
            env = gym.make("CartPole-v0")

        self.env = env
        # Enable video recording features
        self.metadata = self.env.metadata

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._done = True

    def reset(self):
        # assert self._done, "procgen envs cannot be early-restarted"
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._done = done
        return obs, rew, done, info

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __repr__(self):
        return self.env.__repr()

    @property
    def spec(self):
        return self.env.spec

# Register Env in Ray
registry.register_env(
    "SafetyGymWrapper",
    lambda config: SafetyGymWrapper(config)
)
