import os
import ray
from ray import tune
import argparse
import yaml
import importlib
import copy
from ray.tune.utils import merge_dicts
import pickle

import collections
import json
from pathlib import Path
import shelve

import gym
import gym.wrappers
import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from ppo import StateDangerPPOTrainer, ActionDangerPPOTrainer
from ray.rllib.agents.ppo.ppo import PPOTrainer
from utils.loader import load_envs, load_models, load_algorithms

args = argparse.ArgumentParser()
args.add_argument("--baseline", action="store_true", help="whether to use the baseline ppo method")
# args.add_argument("--config", help="the config file path")
# args.add_argument("--tune-config", help="config for hyperparamter tuning")
# args.add_argument("--num-tune-runs", type=int, default=20, help="number of hyperparamter sets to test")
# args.add_argument("--state-danger", action="store_true", help="whether to calculate danger level for states instead of state-action pairs, default is state-action pairs")
# args.add_argument("--callback", action="store_true")
# args.add_argument("--visual-obs", action="store_true", help="whether the observation is visual (an image) or non visual (a vector), default is non visual")
# args.add_argument("--level-file", default=None, help="path to level file")
args.add_argument("--checkpoint", help="path to checkpoint file")
args.add_argument("--num-levels", default=5, type=int)
args.add_argument("--out")
args.add_argument("--video-dir", default=None)
args.add_argument("--explore", type=int)

args = args.parse_args()

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done])
        self._total_steps += 1

def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            video_dir=None):
    policy_agent_mapping = default_policy_agent_mapping

    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env.metadata["render.modes"] = ["human", "rgb_array"]
        env = gym.wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        episode_steps = 0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            episode_steps += 1
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {} steps: {}".format(episodes, reward_total, episode_steps))
        if done:
            episodes += 1



load_envs(os.getcwd()) # Load envs
load_models(os.getcwd()) # Load models

ray.init()

# checkpoint_freq = config.pop("checkpoint_freq", 0)
# checkpoint_at_end = config.pop("checkpoint_at_end", False)
# keep_checkpoints_num = config.pop("keep_checkpoints_num", None)

trainer = None
# cwd = os.path.dirname(os.path.realpath(__file__))

# if args.visual_obs:
#     config["model"]["custom_model"] = "vision_net"
# else:
#     config["model"]["custom_model"] = "simple_fcnet"

# if args.level_file is not None:
#     config["env_config"]["level_file"] = os.path.join(cwd, args.level_file)



config = {}
# Load configuration from checkpoint file.
config_dir = os.path.dirname(args.checkpoint)
config_path = os.path.join(config_dir, "params.pkl")
# Try parent directory.
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

# Set num_workers to be at least 2.
if "num_workers" in config:
    config["num_workers"] = min(2, config["num_workers"])

# Merge with `evaluation_config`.
evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
config = merge_dicts(config, evaluation_config)

if args.video_dir:
    config["env_config"]["render_mode"] = "rgb_array"
args_dict = vars(args)
if args_dict["num_levels"] is not None:
    config["env_config"]["num_levels"] = args_dict["num_levels"]
if args_dict["explore"] is not None:
    config["explore"] = args_dict["explore"]==1

config["evaluation_interval"] = 0
config["monitor"] = False

print(config)

if args.baseline:
    trainer = PPOTrainer
else:
    state_danger = config["model"]["custom_options"]["state_danger"]
    if not state_danger:
        trainer = ActionDangerPPOTrainer
    else:
        trainer = StateDangerPPOTrainer


restored_trainer = trainer(env=config["env"], config=config)
restored_trainer.restore(args.checkpoint)

num_steps = None
num_levels = args.num_levels

# Do the actual rollout.
with RolloutSaver(
        args.out,
        False,
        write_update_file=False,
        target_steps=num_steps,
        target_episodes=num_levels,
        save_info=False) as saver:
    rollout(restored_trainer, config["env"], num_steps, num_levels, saver, True, args.video_dir)