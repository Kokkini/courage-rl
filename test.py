import dangerous_maze_env
import numpy as np

env = dangerous_maze_env.DangerousMazeEnv()

for i in range(10000):
    state, reward, done, info = env.step(np.random.randint(4))
    print(reward)
    env.render()
    if done:
        env.reset()
        env.render()
