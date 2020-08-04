from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt.pyll.base import scope
import numpy as np

space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-3)),
        "vf_clip_param": hp.loguniform("vf_clip_param", np.log(0.04), np.log(1)),
		"entropy_coeff": hp.loguniform("entropy_coeff", np.log(1e-3), np.log(1e-1)),
		"vf_loss_coeff": hp.loguniform("vf_loss_coeff", np.log(0.1), np.log(2.5)),
		"num_sgd_iter": scope.int(hp.quniform("num_sgd_iter", 2, 6, 1))
    }
current_best_params = [
    {
        "lr": 0.0006,
        "vf_clip_param": 0.1,
        "entropy_coeff": 0.0019,
        "num_sgd_iter": 2,
        "vf_loss_coeff": 0.45
    }
]

algo = HyperOptSearch(
    space,
    n_initial_points=10,
    points_to_evaluate=current_best_params
)