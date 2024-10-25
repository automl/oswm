import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cat_encoder_generator_generator(hps, target):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNCatClass(nn.Module):
        def __init__(self, num_features, emsize, replace_nan_by_zero=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_features = num_features
            self.emsize = emsize
            self.replace_nan_by_zero = replace_nan_by_zero
            self.residual_flag = hps["encoder_res_connection"]
            self.target = target
            if self.target:
                self.part_two_in = 1
            else:
                self.part_two_in = 3

            num_hidden = hps["encoder_depth"]
            width_hidden = hps["encoder_width"]
            use_bias = hps["encoder_use_bias"]

            # First Part state or next state
            self.in_lin_1 = torch.nn.Linear(11, width_hidden, bias=use_bias)
            self.in_act_1 = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_1 = nn.ModuleList(layer_list)
            self.out_lin_1 = torch.nn.Linear(width_hidden, emsize//4, bias=use_bias)

            # Second Part action or reward
            self.in_lin_2 = torch.nn.Linear(self.part_two_in, width_hidden, bias=use_bias)
            self.in_act_2 = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_2 = nn.ModuleList(layer_list)
            self.out_lin_2 = torch.nn.Linear(width_hidden, emsize//4, bias=use_bias)

        def forward(self, x):
            if self.replace_nan_by_zero:
                x = torch.nan_to_num(x, nan=0.0)

            residual = self.in_lin_1(x[:, :, :11])
            out = self.in_act_1(residual)
            for layer in self.layer_list_1:
                out = layer(out) + self.residual_flag * residual
            out_1 = self.out_lin_1(out)

            residual = self.in_lin_2(x[:, :, self.num_features-self.part_two_in:])
            out = self.in_act_2(residual)
            for layer in self.layer_list_2:
                out = layer(out) + self.residual_flag * residual
            out_2 = self.out_lin_2(out)
            if self.target:
                zero_pad = torch.full((x.shape[0], x.shape[1], self.emsize // 2), 0., device=device)
                return torch.cat((zero_pad, out_1, out_2), dim=2)
            else:
                zero_pad = torch.full((x.shape[0], x.shape[1], self.emsize // 2), 0., device=device)
                return torch.cat((out_1, out_2, zero_pad), dim=2)

        def __setstate__(self, state):
            super().__setstate__(state)
            self.__dict__.setdefault("replace_nan_by_zero", True)

    return NNCatClass


def mlp_encoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNClass(nn.Module):
        def __init__(self, num_features, emsize, replace_nan_by_zero=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_features = num_features
            self.emsize = emsize
            self.replace_nan_by_zero = replace_nan_by_zero
            num_hidden = hps["encoder_depth"]
            width_hidden = hps["encoder_width"]
            use_bias = hps["encoder_use_bias"]

            self.residual_flag = hps["encoder_res_connection"]

            self.in_lin = torch.nn.Linear(num_features, width_hidden, bias=use_bias)
            self.in_act = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list = nn.ModuleList(layer_list)
            self.out_lin = torch.nn.Linear(width_hidden, emsize, bias=use_bias)

        def forward(self, x):
            if self.replace_nan_by_zero:
                x = torch.nan_to_num(x, nan=0.0)
            residual = self.in_lin(x)
            out = self.in_act(residual)
            for layer in self.layer_list:
                out = layer(out) + self.residual_flag * residual
            return self.out_lin(out)

        def __setstate__(self, state):
            super().__setstate__(state)
            self.__dict__.setdefault("replace_nan_by_zero", True)

    return NNClass


def mlp_decoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNDecClass(nn.Module):
        def __init__(self, ninp, nhid, nout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            num_hidden = hps["decoder_depth"]
            width_hidden = hps["decoder_width"]
            use_bias = hps["decoder_use_bias"]

            self.residual_flag = hps["decoder_res_connection"]

            self.in_lin = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list = nn.ModuleList(layer_list)
            self.out_lin = torch.nn.Linear(width_hidden, nout, bias=use_bias)

        def forward(self, x):
            residual = self.in_lin(x)
            out = self.in_act(residual)
            for layer in self.layer_list:
                out = layer(out) + self.residual_flag * residual
            return self.out_lin(out)

    return NNDecClass


def cat_decoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNCatDecClass(nn.Module):
        def __init__(self, ninp, nhid, nout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            num_hidden = hps["decoder_depth"]
            width_hidden = hps["decoder_width"]
            use_bias = hps["decoder_use_bias"]

            self.residual_flag = hps["decoder_res_connection"]

            # Decodes state
            self.in_lin_1 = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act_1 = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_1 = nn.ModuleList(layer_list)
            self.out_lin_1 = torch.nn.Linear(width_hidden, nout-3, bias=use_bias)

            # Decodes Reward
            self.in_lin_2 = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act_2 = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_2 = nn.ModuleList(layer_list)
            self.out_lin_2 = torch.nn.Linear(width_hidden, 1, bias=use_bias)

        def forward(self, x):
            residual = self.in_lin_1(x)
            out = self.in_act_1(residual)
            for layer in self.layer_list_1:
                out = layer(out) + self.residual_flag * residual
            out_1 = self.out_lin_1(out)

            residual = self.in_lin_2(x)
            out = self.in_act_2(residual)
            for layer in self.layer_list_2:
                out = layer(out) + self.residual_flag * residual
            out_2 = self.out_lin_2(out)

            zero_shape = (x.shape[0], x.shape[1], 2)
            zero_padding = torch.full(zero_shape, 0., device=device)
            return torch.cat((out_1, zero_padding, out_2), dim=2)

    return NNCatDecClass


if __name__ == '__main__':
    print("nothin to see")