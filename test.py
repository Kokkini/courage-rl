from envs.dangerous_maze_env import DangerousMazeEnv
import numpy as np

config = {"level_file": "envs/6x6.txt", "death_penalty": True}
env = DangerousMazeEnv(config)

for i in range(10000):
    state, reward, done, info = env.step(np.random.randint(4))
    print(reward)
    env.render()
    if done:
        env.reset()
        env.render()
