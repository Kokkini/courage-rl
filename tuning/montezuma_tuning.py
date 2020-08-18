from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "gamma": hp.uniform("gamma", 0.9, 0.999),
        "danger_loss_coeff": hp.loguniform("danger_loss_coeff", np.log(1e-2), np.log(1)),
        "entropy_coeff": hp.loguniform("entropy_coeff", np.log(1e-2), np.log(1e-1)),
        "danger_reward_coeff": hp.loguniform("danger_reward_coeff", np.log(1e-2), np.log(1)),
        "gamma_death": hp.uniform("gamma_death", 0.9, 0.999),
        "death_reward": hp.uniform("death_reward", -1, 0),
    }
current_best_params = []

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)