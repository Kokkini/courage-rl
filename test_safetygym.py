# from envs.dangerous_maze_env import DangerousMazeEnv
# import numpy as np
# from PIL import Image
#
# config = {"level_file": "envs/6x6.txt", "death_penalty": True}
# env = DangerousMazeEnv(config)
# img = env.render()
# img = Image.fromarray(img)
# img.save("media/6x6.jpg")
# for i in range(10000):
#     state, reward, done, info = env.step(np.random.randint(4))
#     print(reward)
#     env.render()
#     if done:
#         env.reset()
#         env.render()

import numpy as np
from envs.safety_gym_wrapper import SafetyGymWrapper
import yaml
import gym
print(gym.__path__)

# config = {
#     'robot_base': 'xmls/car.xml',
#     'num_steps': 3000,
#     'robot_locations': [(-1,0)],
#     'robot_rot': np.pi,
#     'task': 'goal',
#     'observe_goal_lidar': True,
#     'observe_box_lidar': True,
#     'observe_hazards': True,
#     'constrain_hazards': True,
#     'lidar_max_dist': 3,
#     'lidar_num_bins': 16,
#     'hazards_num': 22,
#     'randomize_layout': True,
#     'hazards_locations': [(-2,-1),(-2,0),(-2,1),(-2,2),(-2,3),(-1,-1),(-1,3),(0,-1),(0,0),(0,1),(0,3),(1,-1),(1,0),(1,1),(1,3),(2,-1),(2,3),(3,-1),(3,0),(3,1),(3,2),(3,3)],
#     'hazards_size': 0.5,
#     'hazards_keepout': 0.01,
#     'goal_locations': [(2,0)],
#     'goal_size': 0.5,
#     'hazards_cost': 1.0,
# }
with open("experiments/road_width_baseline.txt") as f:
    config = yaml.safe_load(f)
    

    # contents = f.read()
    # config = ast.literal_eval(contents)
    level_file = "envs/safety_gym_narrow_road_s_eval.txt"
    config["env_config"]["level_file"] = level_file
    config = config["env_config"]

print(config)
env = SafetyGymWrapper(config)
print(env.env.spec)
print(env.env.__dict__)
print(env.action_space)
env.reset()

while True:
    action = env.action_space.sample()
    # print("action:", action)
    next_observation, reward, done, info = env.step(action)
    img = env.render(mode="human")#.astype(np.uint8)
    # print(img)
    # print(img.shape)
    # print(type(img))
    # print(img.dtype)
    # print(np.dtype(img))
    if done:
        env.reset()