import logging

import ray
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule

from postprocessing import compute_advantages, compute_advantages_and_danger, Postprocessing
from sample_batch import SampleBatch


print("LOADED CUSTOM POLICY")

tf = try_import_tf()

logger = logging.getLogger(__name__)


class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 danger_targets,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 danger_preds,
                 curr_action_dist,
                 value_fn,
                 danger_fn,
                 encoding_fn,
                 encoding_targets,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 danger_loss_coeff=1.0,
                 curiosity_loss_coeff=1.0,
                 use_curiosity=False,
                 use_gae=True,
                 action_danger=False):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for action prob output
                from the previous (before update) Model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from the previous (before update) Model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Optional[tf.Tensor]): An optional bool mask of valid
                input elements (for max-len padded sequences (RNNs)).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:

            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        else:

            def reduce_mean_valid(t):
                return tf.reduce_mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if action_danger:
            action_onehot = tf.one_hot(actions, tf.shape(prev_logits)[1])
            danger_fn = tf.reduce_sum(danger_fn*action_onehot, axis=1)

        danger_loss1 = tf.square(danger_fn - danger_targets)
        danger_clipped = danger_preds + tf.clip_by_value(danger_fn - danger_preds, -vf_clip_param, vf_clip_param)
        danger_loss2 = tf.square(danger_clipped - danger_targets)
        danger_loss = tf.maximum(danger_loss1, danger_loss2)
        self.mean_danger_loss = reduce_mean_valid(danger_loss)
        self.danger_fn = danger_fn

        self.mean_curiosity_loss = 0
        curiosity_loss = 0
        if use_curiosity:
            curiosity_loss = tf.reduce_mean(tf.square(encoding_fn - encoding_targets), axis=1)
            self.mean_curiosity_loss = reduce_mean_valid(curiosity_loss)


        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss + danger_loss_coeff * danger_loss -
                entropy_coeff * curr_entropy + curiosity_loss * curiosity_loss_coeff)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss

def get_loss(action_danger=False):
    def ppo_surrogate_loss(policy, model, dist_class, train_batch):
        logits, state = model.from_batch(train_batch)
        action_dist = dist_class(logits, model)

        mask = None
        if state:
            max_seq_len = tf.reduce_max(train_batch["seq_lens"])
            mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
            mask = tf.reshape(mask, [-1])

        policy.loss_obj = PPOLoss(
            dist_class=dist_class,
            model=model,
            value_targets=train_batch[Postprocessing.VALUE_TARGETS],
            advantages=train_batch[Postprocessing.ADVANTAGES],
            danger_targets=train_batch[Postprocessing.DANGER_TARGETS],
            actions=train_batch[SampleBatch.ACTIONS],
            prev_logits=train_batch[SampleBatch.ACTION_DIST_INPUTS],
            prev_actions_logp=train_batch[SampleBatch.ACTION_LOGP],
            vf_preds=train_batch[SampleBatch.VF_PREDS],
            danger_preds=train_batch[SampleBatch.DANGER_PREDS],
            curr_action_dist=action_dist,
            value_fn=model.value_function(),
            danger_fn=model.danger_score_function(),
            encoding_fn=model.get_encoding(),
            encoding_targets=train_batch[SampleBatch.ENCODING_RANDOM],
            cur_kl_coeff=policy.kl_coeff,
            valid_mask=mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"],
            vf_clip_param=policy.config["vf_clip_param"],
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            danger_loss_coeff=policy.config["danger_loss_coeff"],
            curiosity_loss_coeff=policy.config["curiosity_loss_coeff"],
            use_curiosity=policy.config["use_curiosity"],
            use_gae=policy.config["use_gae"],
            action_danger=action_danger
        )
        return policy.loss_obj.loss
    return ppo_surrogate_loss

def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "danger_loss": policy.loss_obj.mean_danger_loss,
        "cur_danger_reward_coeff": tf.cast(policy.danger_reward_coeff, tf.float64),
        "cur_ext_reward_coeff": tf.cast(policy.ext_reward_coeff, tf.float64),
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "danger_explained_var": explained_variance(
            train_batch[Postprocessing.DANGER_TARGETS],
            policy.loss_obj.danger_fn),
        "mean_danger_targets": tf.reduce_mean(
            train_batch[Postprocessing.DANGER_TARGETS]),
        "mean_danger": tf.reduce_mean(
            policy.loss_obj.danger_fn),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


def vf_preds_fetches(policy):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }


def value_and_danger_fetches(policy):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        SampleBatch.DANGER_PREDS: policy.model.danger_score_function(),
        SampleBatch.ENCODING: policy.model.get_encoding(),
        SampleBatch.ENCODING_RANDOM: policy.model.get_encoding_random(),
        SampleBatch.DANGER_REWARD_COEFF: tf.zeros(tf.shape(policy.model.value_function())[0]) + tf.cast(policy.danger_reward_coeff, tf.float32),
        SampleBatch.EXT_REWARD_COEFF: tf.zeros(tf.shape(policy.model.value_function())[0]) + tf.cast(policy.ext_reward_coeff, tf.float32)
    }

def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        if len(sample_batch[SampleBatch.DONES]) > 1:  # rllib uses a dummy trajectory of length 1 to initialize the loss
            print("A trajectory did not completed. Only support complete trajectories")
            exit(1)

        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    # batch = compute_advantages_and_danger(
    #     sample_batch,
    #     last_r,
    #     gamma=policy.config["gamma"],
    #     lambda_=policy.config["lambda"],
    #     lambda_death=policy.config["lambda_death"],
    #     gamma_death=policy.config["gamma_death"],
    #     danger_reward_coeff=policy.config["danger_reward_coeff"],
    #     death_reward=policy.config["death_reward"],
    #     env_max_step=policy.config["max_step"],
    #     use_gae=policy.config["use_gae"],
    # )
    batch = compute_advantages_and_danger(
        sample_batch,
        last_r,
        policy.config
    )
    return batch


def clip_gradients(policy, optimizer, loss):
    variables = policy.model.trainable_variables()
    if policy.config["grad_clip"] is not None:
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        grads = [g for (g, v) in grads_and_vars]
        policy.grads, _ = tf.clip_by_global_norm(grads,
                                                 policy.config["grad_clip"])
        clipped_grads = list(zip(policy.grads, variables))
        return clipped_grads
    else:
        return optimizer.compute_gradients(loss, variables)


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff_val *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff_val *= 0.5
        self.kl_coeff.load(self.kl_coeff_val, session=self.get_session())
        return self.kl_coeff_val


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        [prev_reward]),
                    "is_training": tf.convert_to_tensor([False]),
                }, [tf.convert_to_tensor([s]) for s in state],
                                          tf.convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._value = value


def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


@DeveloperAPI
class DangerRewardCoeffSchedule:
    @DeveloperAPI
    def __init__(self, danger_reward_coeff, danger_reward_coeff_schedule):
        self.danger_reward_coeff = tf.get_variable("danger_reward_coeff", initializer=float(danger_reward_coeff), trainable=False)

        if danger_reward_coeff_schedule is None:
            self.danger_reward_coeff_schedule = ConstantSchedule(
                danger_reward_coeff, framework=None)
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(danger_reward_coeff_schedule, list):
                self.danger_reward_coeff_schedule = PiecewiseSchedule(
                    danger_reward_coeff_schedule,
                    outside_value=danger_reward_coeff_schedule[-1][-1],
                    framework=None)
            else:
                # Implements previous version but enforces outside_value
                self.danger_reward_coeff_schedule = PiecewiseSchedule(
                    [[0, danger_reward_coeff], [danger_reward_coeff_schedule, 0.0]],
                    outside_value=0.0,
                    framework=None)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(DangerRewardCoeffSchedule, self).on_global_var_update(global_vars)
        self.danger_reward_coeff.load(
            self.danger_reward_coeff_schedule.value(global_vars["timestep"]),
            session=self._sess)


@DeveloperAPI
class ExtRewardCoeffSchedule:
    @DeveloperAPI
    def __init__(self, ext_reward_coeff, ext_reward_coeff_schedule):
        self.ext_reward_coeff = tf.get_variable("ext_reward_coeff", initializer=float(ext_reward_coeff), trainable=False)

        if ext_reward_coeff_schedule is None:
            self.ext_reward_coeff_schedule = ConstantSchedule(
                ext_reward_coeff, framework=None)
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(ext_reward_coeff_schedule, list):
                self.ext_reward_coeff_schedule = PiecewiseSchedule(
                    ext_reward_coeff_schedule,
                    outside_value=ext_reward_coeff_schedule[-1][-1],
                    framework=None)
            else:
                # Implements previous version but enforces outside_value
                self.ext_reward_coeff_schedule = PiecewiseSchedule(
                    [[0, ext_reward_coeff], [ext_reward_coeff_schedule, 0.0]],
                    outside_value=0.0,
                    framework=None)

    @override(Policy)
    def on_global_var_update(self, global_vars):
        super(ExtRewardCoeffSchedule, self).on_global_var_update(global_vars)
        self.ext_reward_coeff.load(
            self.ext_reward_coeff_schedule.value(global_vars["timestep"]),
            session=self._sess)


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    DangerRewardCoeffSchedule.__init__(policy, config["danger_reward_coeff"], config["danger_reward_coeff_schedule"])
    ExtRewardCoeffSchedule.__init__(policy, config["ext_reward_coeff"], config["ext_reward_coeff_schedule"])


PPOTFStateDangerPolicy = build_tf_policy(
    name="PPOTFStateDangerPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=get_loss(action_danger=False),
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=value_and_danger_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, DangerRewardCoeffSchedule, ExtRewardCoeffSchedule
    ])

PPOTFActionDangerPolicy = build_tf_policy(
    name="PPOTFActionDangerPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=get_loss(action_danger=True),
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=value_and_danger_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, DangerRewardCoeffSchedule, ExtRewardCoeffSchedule
    ])
