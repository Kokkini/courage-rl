import numpy as np
import scipy.signal
from ray.rllib.utils.annotations import DeveloperAPI

from sample_batch import SampleBatch

print("LOADED CUSTOM POSTPROCESSING")

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    DANGER_TARGETS = "danger_targets"
    DANGER_REWARD = "danger_reward"
    CURIOSITY_REWARD = "curiosity_reward"

def get_one_each_row(arr, indices):
    return np.array([arr[enum, item] for enum, item in enumerate(indices)])

@DeveloperAPI
def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


# @DeveloperAPI
# def compute_advantages_and_danger(rollout,last_r,gamma=0.99,lambda_=1.0,lambda_death=1.0, gamma_death=0.0, danger_reward_coeff=1, death_reward=0, env_max_step=1000, use_gae=True,use_critic=True):
#     """
#     Given a rollout, compute its value targets and the advantage.
#
#     Args:
#         rollout (SampleBatch): SampleBatch of a single trajectory
#         last_r (float): Value estimation for last observation
#         gamma (float): Discount factor.
#         lambda_ (float): Parameter for GAE
#         use_gae (bool): Using Generalized Advantage Estimation
#         use_critic (bool): Whether to use critic (value estimates). Setting
#                            this to False will use 0 as baseline.
#
#     Returns:
#         SampleBatch (SampleBatch): Object with experience from rollout and
#             processed rewards.
#     """
#
#     traj = {}
#     trajsize = len(rollout[SampleBatch.ACTIONS])
#     for key in rollout:
#         traj[key] = np.stack(rollout[key])
#
#     assert SampleBatch.VF_PREDS in rollout or not use_critic, \
#         "use_critic=True but values not found"
#     assert use_critic or not use_gae, \
#         "Can't use gae without using a value function"
#
#     death = np.zeros(trajsize, dtype=np.float32)
#
#     if len(traj[SampleBatch.DANGER_PREDS].shape) > 1:
#         traj[SampleBatch.DANGER_PREDS] = get_one_each_row(traj[SampleBatch.DANGER_PREDS], traj[SampleBatch.ACTIONS])
#     traj[Postprocessing.DANGER_REWARD] = traj[SampleBatch.DANGER_PREDS].copy()
#     if trajsize < env_max_step and traj[SampleBatch.REWARDS][-1] <= 0: # it died
#         traj[Postprocessing.DANGER_REWARD][-1] = death_reward
#         death[-1] = 1.0
#
#     traj[SampleBatch.REWARDS] = traj[SampleBatch.REWARDS].astype(np.float32)
#     traj[SampleBatch.REWARDS] += traj[Postprocessing.DANGER_REWARD] * danger_reward_coeff
#
#
#     danger_pred_t = np.concatenate(
#         [traj[SampleBatch.DANGER_PREDS],
#          np.array([0.0])])
#     danger_delta_t = (
#         death + gamma_death * danger_pred_t[1:] - danger_pred_t[:-1])
#     danger_advantage = discount(danger_delta_t, gamma_death * lambda_death)
#     traj[Postprocessing.DANGER_TARGETS] = (
#         danger_advantage + traj[SampleBatch.DANGER_PREDS]).copy().astype(np.float32)
#
#     if use_gae:
#         vpred_t = np.concatenate(
#             [rollout[SampleBatch.VF_PREDS],
#              np.array([last_r])])
#         delta_t = (
#             traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
#         # This formula for the advantage comes from:
#         # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
#         traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
#         traj[Postprocessing.VALUE_TARGETS] = (
#             traj[Postprocessing.ADVANTAGES] +
#             traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
#     else:
#         rewards_plus_v = np.concatenate(
#             [rollout[SampleBatch.REWARDS],
#              np.array([last_r])])
#         discounted_returns = discount(rewards_plus_v,
#                                       gamma)[:-1].copy().astype(np.float32)
#
#         if use_critic:
#             traj[Postprocessing.
#                  ADVANTAGES] = discounted_returns - rollout[SampleBatch.
#                                                             VF_PREDS]
#             traj[Postprocessing.VALUE_TARGETS] = discounted_returns
#         else:
#             traj[Postprocessing.ADVANTAGES] = discounted_returns
#             traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
#                 traj[Postprocessing.ADVANTAGES])
#
#     traj[Postprocessing.ADVANTAGES] = traj[
#         Postprocessing.ADVANTAGES].copy().astype(np.float32)
#
#     assert all(val.shape[0] == trajsize for val in traj.values()), \
#         "Rollout stacked incorrectly!"
#     return SampleBatch(traj)

@DeveloperAPI
def compute_advantages_and_danger(rollout, last_r, config, use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """
    gamma = config["gamma"]
    lambda_= config["lambda"]
    lambda_death = config["lambda_death"]
    gamma_death = config["gamma_death"]
    danger_reward_coeff = config["danger_reward_coeff"]
    death_reward = config["death_reward"]
    env_max_step = config["max_step"]
    use_gae = config["use_gae"]
    reward_discount_on_death = config["reward_discount_on_death"]
    use_death_reward = config["use_death_reward"]
    curiosity_reward_coeff = config["curiosity_reward_coeff"]
    use_curiosity = config["use_curiosity"]

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    death = np.zeros(trajsize, dtype=np.float32)

    if len(traj[SampleBatch.DANGER_PREDS].shape) > 1: 
        traj[SampleBatch.DANGER_PREDS] = get_one_each_row(traj[SampleBatch.DANGER_PREDS], traj[SampleBatch.ACTIONS])
        traj[Postprocessing.DANGER_REWARD] = traj[SampleBatch.DANGER_PREDS].copy()
    else:
        temp = traj[SampleBatch.DANGER_PREDS].copy()
        temp = np.roll(temp, -1)
        temp[-1] = 0
        traj[Postprocessing.DANGER_REWARD] = temp


    
    if trajsize < env_max_step and traj[SampleBatch.REWARDS][-1] <= 0: # it died
        if use_death_reward:
            traj[Postprocessing.DANGER_REWARD][-1] = death_reward
        else:
            discounts = 1 - reward_discount_on_death ** np.arange(trajsize)
            discounts = np.flip(discounts, axis=0)
            traj[Postprocessing.DANGER_REWARD] *= discounts
        death[-1] = 1.0

    curiosity_reward = np.array([0]*trajsize)
    if use_curiosity:
        curiosity_reward = traj[SampleBatch.ENCODING] - traj[SampleBatch.ENCODING_RANDOM]
        curiosity_reward = np.sqrt(np.mean(np.square(curiosity_reward), axis=1))
        curiosity_reward = np.roll(curiosity_reward, -1)
        curiosity_reward[-1] = 0

    traj[Postprocessing.CURIOSITY_REWARD] = curiosity_reward

    traj[SampleBatch.REWARDS] = traj[SampleBatch.REWARDS].astype(np.float32)
    traj[SampleBatch.REWARDS] += traj[Postprocessing.DANGER_REWARD] * danger_reward_coeff + traj[Postprocessing.CURIOSITY_REWARD] * curiosity_reward_coeff


    danger_pred_t = np.concatenate(
        [traj[SampleBatch.DANGER_PREDS],
         np.array([0.0])])
    danger_delta_t = (
        death + gamma_death * danger_pred_t[1:] - danger_pred_t[:-1])
    danger_advantage = discount(danger_delta_t, gamma_death * lambda_death)
    traj[Postprocessing.DANGER_TARGETS] = (
        danger_advantage + traj[SampleBatch.DANGER_PREDS]).copy().astype(np.float32)

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)
