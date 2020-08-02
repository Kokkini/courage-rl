import ray
from ray import tune
from ray.rllib.agents.dqn.dqn_tf_policy import *
from ray.rllib.agents.dqn.dqn import *
import dangerous_maze_env

DQN_Policy = build_tf_policy(
    name="DQN_Policy",
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model=build_q_model,
    action_distribution_fn=get_distribution_inputs_and_class,
    loss_fn=build_q_losses,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    gradients_fn=clip_gradients,
    extra_action_fetches_fn=lambda policy: {"q_values": policy.q_values},
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    before_init=setup_early_mixins,
    before_loss_init=setup_mid_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])

DQN_Trainer = build_trainer(
    name="DQN_Trainer",
    default_policy=DQN_Policy,
    get_policy_class=get_policy_class,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    get_initial_state=get_initial_state,
    make_policy_optimizer=make_policy_optimizer,
    before_train_step=update_worker_exploration,
    after_optimizer_step=update_target_if_needed,
    after_train_result=after_train_result,
    execution_plan=execution_plan)

ray.init()
print(DEFAULT_CONFIG)
config={
  "env": dangerous_maze_env.DangerousMazeEnv, 
  "num_workers": 2,
  "num_cpus_per_worker": 0.5,
  "model": {"conv_filters": [[32, [3, 3], 2], [64, [3, 3], 1], [128, [5, 5], 1]]}}
tune.run(DQN_Trainer, config=config)
