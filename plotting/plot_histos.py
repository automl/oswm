from priors import rl_prior
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

hps = {"env_name": "MomentumEnv", "num_hidden": 1, "relu": False, "sigmoid": False, "sin": True,
       "state_offset": 3.2802608490289904, "state_scale": 18.147661409701062, "tanh": True, "test": False,
       "use_bias": False, "use_dropout": False, "use_layer_norm": True, "use_res_connection": True, "width_hidden": 16,
       "no_norm": False}

batch = rl_prior.get_batch(8, 1001, 14, hyperparameters=hps)
matplotlib.rcParams.update({'font.size': 14})

print(batch.x.shape)
for b in range(batch.x.shape[1]):
    for dim in range(batch.x.shape[2]-3):
        data = batch.x[:, b, dim]
        minimum = data.min().item()
        maximum = data.max().item()
        if minimum == maximum:
            continue
        counts, bins = np.histogram(data, bins=100, range=(minimum, maximum))
        plt.stairs(counts, bins, alpha=1., fill=True, label="Distribution of samples in prior", color="steelblue")
        plt.title(f"Distribution of samples of prior")
        plt.xlabel("Values")
        plt.ylabel("Counts")
        plt.savefig(f"../prior_histos/test_b_{b}_dim_{dim}.png")
        plt.clf()
