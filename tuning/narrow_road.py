import numpy as np

config_list = []

# vals = [[0.9,0.3]]
vals = [[0,0]]

for i in range(len(vals)):
    rescale = 1 / 10
    config = {
            # "lr": float(np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))),
            # "gamma": float(np.random.uniform(0.9, 0.999)),
            # "danger_loss_coeff": hp.loguniform("danger_loss_coeff", np.log(1e-2), np.log(1)),
            # "entropy_coeff": hp.loguniform("entropy_coeff", np.log(1e-2), np.log(1e-1)),
            # "danger_reward_coeff": hp.loguniform("danger_reward_coeff", np.log(1e-2), np.log(1)),
            # "gamma_death": hp.uniform("gamma_death", 0.9, 0.999),
            # "death_reward": hp.uniform("death_reward", -1, 0),
            "danger_reward_coeff": vals[i][0] * rescale,
            "danger_reward_coeff_schedule": [[0,vals[i][0] * rescale],[400000,vals[i][1] * rescale]],

            "use_death_reward": True,
            "death_reward": float(-1),

            # "period": vals[i],
            "ext_reward_coeff": 1 - vals[i][0],
            "ext_reward_coeff_schedule": [[0,1-vals[i][0]],[400000,1-vals[i][1]]],
            
        }
    config_list.append(config)

# current_best_params = []

# algo = HyperOptSearch(
#     space,
#     n_initial_points=10,
#     points_to_evaluate=current_best_params
# )