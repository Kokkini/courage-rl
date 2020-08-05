import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from ray.tune import registry
from PIL import Image


class DangerousMazeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NO_OP = 4
    N_DISCRETE_ACTIONS = 5
    MAX_ITER = 200

    char_map = {
        "p": np.array([0, 0, 255], dtype=np.uint8),
        ".": np.array([0, 0, 0], dtype=np.uint8),
        "x": np.array([255, 0, 0], dtype=np.uint8),
        "g": np.array([0, 255, 0], dtype=np.uint8),
        "G": np.array([255, 255, 0], dtype=np.uint8)
    }

    action_map = {
        UP: np.array([-1, 0], np.int32),
        DOWN: np.array([1, 0], np.int32),
        LEFT: np.array([0, -1], np.int32),
        RIGHT: np.array([0, 1], np.int32),
        NO_OP: np.array([0, 0], np.int32)
    }

    def __init__(self, config=None):
        super(DangerousMazeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        self.string_rep = self.get_string_rep(config["level_file"])
        self.obs_shape = (len(self.string_rep), len(self.string_rep[0]), 3)

        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        # Example for using image as input:

        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.state, self.player_pos = self.state_from_string_rep(self.string_rep)
        self.iter = 0
        if config["death_penalty"]:
            death_reward = -1
        else:
            death_reward = 0
        self.reward_map = {
            "x": death_reward,
            ".": 0,
            "g": 1,
            "G": 10
        }

    def get_string_rep(self, filepath):
        result = []
        with open(filepath) as f:
            for line in f.readlines():
                if len(line) > 1:
                    result.append(line.strip())
                else:
                    print(f"there's an empty line in the string representation file {filepath} of the environment")
                    exit(1)
        return result

    def step(self, action):
        # Execute one time step within the environment
        done = False
        self.iter += 1
        prev_pos = self.player_pos.copy()
        pos = prev_pos + self.action_map[action]
        pos = np.minimum(self.obs_shape[:2] - np.array([1, 1], np.int32), np.maximum(pos, 0))
        self.player_pos = pos
        prev_c = self.get_char(prev_pos, self.string_rep)
        c = self.get_char(pos, self.string_rep)
        # calculate state
        self.state[prev_pos[0], prev_pos[1]] = self.char_map[prev_c]
        self.state[pos[0], pos[1]] = self.mix_color(self.char_map[c], self.char_map["p"])
        state = self.state

        # calculate reward
        reward = self.reward_map[c]

        if c in ["x", "g", "G"]:
            done = True
        if self.iter >= self.MAX_ITER:
            done = True
        info = {}

        return state, reward, done, info

    def get_char(self, pos, string_rep):
        c = string_rep[pos[0]][pos[1]]
        if c == "p":
            c = "."
        return c

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state, self.player_pos = self.state_from_string_rep(self.string_rep)
        self.iter = 0
        return self.state

    def render(self, mode='rgb_array', close=False):
        # Render the environment to the screen
        if mode == "human":
            plt.imshow(self.state)
            plt.pause(0.01)

        enlarge_factor = 30
        img = Image.fromarray(self.state)
        new_size = (self.state.shape[0] * enlarge_factor, self.state.shape[1] * enlarge_factor)
        img = img.resize(new_size, Image.NEAREST)
        # img = np.array(img)
        # print(img.shape)
        # img = self.scale(self.state, new_size[0], new_size[1])
        return np.array(img, np.uint8)

    def state_from_string_rep(self, string_rep):
        state = []
        player_pos = None
        for r in range(len(string_rep)):
            state.append([])
            for c in range(len(string_rep[0])):
                if string_rep[r][c] == "p":
                    player_pos = np.array([r, c], dtype=np.int32)
                state[r].append(self.char_map[string_rep[r][c]])
        state = np.array(state, dtype=np.uint8)
        return state, player_pos

    def mix_color(self, c1, c2):
        c1 = np.array(c1, np.float32)
        c2 = np.array(c2, np.float32)
        c = c1 + c2
        c = np.floor(c / np.max(c) * 255)
        c = np.array(c, np.uint8)
        return c

    def scale(self, im, nR, nC):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns
        return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                 for c in range(nC)] for r in range(nR)]


# Register Env in Ray
registry.register_env(
    "DangerousMaze",
    lambda config: DangerousMazeEnv(config)
)