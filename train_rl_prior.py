import sys
sys.path.insert(0, '..')

import torch
from torch import nn
import numpy as np
import random
from train import train
import priors.rl_prior
import encoders
import utils
from decoder import *
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# maximum Dimension of observations + max dimension of action
num_features = 14

encoder_decoder_hps = {"decoder_activation": "sigmoid", "decoder_depth": 2, "decoder_res_connection": True,
                       "decoder_type": "cat", "decoder_use_bias": False, "decoder_width": 64,
                       "encoder_activation": "gelu", "encoder_depth": 3,
                       "encoder_res_connection": True, "encoder_type": "cat", "encoder_use_bias": True,
                       "encoder_width": 512}

criterion = nn.MSELoss(reduction='none')

if encoder_decoder_hps["encoder_type"] == "mlp":
    gen_x = mlp_encoder_generator_generator(encoder_decoder_hps)
    gen_y = gen_x
elif encoder_decoder_hps["encoder_type"] == "cat":
    gen_x = cat_encoder_generator_generator(encoder_decoder_hps, target=False)
    gen_y = cat_encoder_generator_generator(encoder_decoder_hps, target=True)

if encoder_decoder_hps["encoder_type"] == "mlp":
    dec_model = mlp_decoder_generator_generator(encoder_decoder_hps)
elif encoder_decoder_hps["decoder_type"] == "cat":
    dec_model = cat_decoder_generator_generator(encoder_decoder_hps)

decoder_dict = {"standard": (dec_model, 14)}

# number of data points provided at train time
train_len = 1000
min_train_len = 500

max_dataset_size = 1001
epochs = 50

# PRIOR hps
hps = {"env_name": "MomentumEnv", "num_hidden": 1, "relu": False, "sigmoid": False, "sin": True,
       "state_offset": 3.2802608490289904, "state_scale": 18.147661409701062, "tanh": True, "test": False,
       "use_bias": False, "use_dropout": False, "use_layer_norm": True, "use_res_connection": True,
       "width_hidden": 16, "no_norm": False, "max_num_state": 11, "max_num_action": 3}

train_result = train(get_batch_method=priors.rl_prior.get_batch, criterion=criterion,
                     # define the transformer size
                     emsize=512, nhead=4, nhid=1024, nlayers=6,
                     # how to encode the x and y inputs to the transformer
                     encoder_generator=gen_x,
                     y_encoder_generator=gen_y,
                     extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
                     epochs=epochs, warmup_epochs=epochs//4, steps_per_epoch=100, batch_size=4,
                     lr=.00005,
                     seq_len=max_dataset_size,
                     single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(train_len, min_train_len, p=0.4),
                     decoder_dict=decoder_dict)

final_mean_loss, final_per_datasetsize_losses, trained_model, dataloader = train_result


torch.save(trained_model.state_dict(), "trained_models/cpkt.pt")
