from typing import Dict
import os
import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
import tensorflow as tf
import numpy as np
from PIL import Image

from sample_batch import SampleBatch

class CustomCallbacks(DefaultCallbacks):
    """
    Please refer to :
        https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    for examples on adding your custom metrics and callbacks.

    This code adapts the documentations of the individual functions from :
    https://github.com/ray-project/ray/blob/master/rllib/agents/callbacks.py

    These callbacks can be used for custom metrics and custom postprocessing.
    """
    ep_rewards = []
    num_eval_eps = 1000
    best_so_far = -float("inf")
    env = None
    img_log_dir = "/content/imgs"
    iter = 0

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        """Callback run on the rollout worker before each episode starts.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        self.env = base_env.get_unwrapped()

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        """Runs on each episode step.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        """Runs when an episode is done.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        ######################################################################
        # An example of adding a custom metric from the latest observation
        # from your env
        ######################################################################
        # last_obs_object_from_episode = episode.last_observation_for()
        # We define a dummy custom metric, observation_mean
        # episode.custom_metrics["observation_mean"] = last_obs_object_from_episode.mean()
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str,
            policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        """Called immediately after a policy's postprocess_fn is called.
        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches (dict): Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        """Called at the end RolloutWorker.sample().
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            samples (SampleBatch): Batch to be returned. You can mutate this
                object to modify the samples generated.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().
        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # In this case we also print the mean timesteps throughput
        # for easier reference in the logs
        # print("=============================================================")
        # print(" Timesteps Throughput : {} ts/sec".format(TBD))
        # print("=============================================================")

        # print("end of Trainable.train()")
        # print(f"result dict: {result}")
        # self.ep_rewards = self.ep_rewards + result["hist_stats"]["episode_reward"]
        # if len(self.ep_rewards) >= self.num_eval_eps:
        #     mean_reward = np.mean(self.ep_rewards[-self.num_eval_eps:])
        #     print(f"mean reward of last {self.num_eval_eps} eps: {mean_reward}")
        #     print(f"best so far: {self.best_so_far}")
        #     if mean_reward > self.best_so_far:
        #         self.best_so_far = mean_reward
        #         trainer.save()
        self.iter += 1
        all_states, base_state = self.env.get_all_states()
        # sample_obs = np.random.randint(0,255,(8*8*3,),np.uint8)
        fetch = trainer.compute_action(all_states, full_fetch=True)
        danger_score = fetch[SampleBatch.DANGER_PREDS]
        img = self.visualize(base_state, danger_score)
        img = Image.fromarray(img)
        os.makedirs(self.img_log_dir, exist_ok=True)
        img.save(os.path.join(self.img_log_dir, f"danger_viz_{self.iter:03}.jpg"))


    def visualize(self, base_state, danger_score):
        # get enlarged image of the base_state
        enlarge_factor = 30
        small_size = enlarge_factor // 3

        img = Image.fromarray(base_state)
        new_size = (base_state.state.shape[0] * enlarge_factor, base_state.state.shape[1] * enlarge_factor)
        img = img.resize(new_size, Image.NEAREST)
        img = np.array(img, np.uint8)

        num_acts = danger_score.shape[-1]
        danger_score = np.reshape(danger_score, (base_state.shape[0], base_state.shape[1], num_acts))

        # visualize the danger score
        lowest = np.array([0,0,255])
        highest = np.array([255,255,0])

        position_mapping = {
            self.env.UP: (0, 1),
            self.env.DOWN: (2, 1),
            self.env.LEFT: (1, 0),
            self.env.RIGHT: (1, 2),
            self.env.NO_OP: (1, 1)
        }

        for row in range(base_state.shape[0]):
            for col in range(base_state.shape[1]):
                anchor = (row*enlarge_factor, col*enlarge_factor)
                for i in range(num_acts):
                    pos = position_mapping[i]
                    score = danger_score[row, col, i]
                    color = np.array(np.clip(highest * score + lowest * (1-score), 0, 255), np.int32)
                    img[anchor[0] + small_size*pos[0], anchor[1]+small_size*pos[1]] = np.tile(color, small_size*small_size).reshape([small_size, small_size, 3])
        return img