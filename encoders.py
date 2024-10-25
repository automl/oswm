import math

import torch
import torch.nn as nn
from utils import normalize_data
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import normalize_data, to_ranking_low_mem, remove_outliers

# TODO: Use something like the Batch Qrapping Sequence


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


class StyleEmbEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size, num_embeddings=100):
        super().__init__()
        assert num_hyperparameters == 1
        self.em_size = em_size
        self.embedding = nn.Embedding(num_embeddings, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters.squeeze(1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # T x B x num_features
        assert self.d_model % x.shape[-1] * 2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (
            (1.0 / interval_size)
            * 2
            * math.pi
            * torch.exp(
                torch.arange(
                    0, d_per_feature, 2, device=self.device_test_tensor.device
                ).float()
                * math.log(math.sqrt(2))
            )
        )
        # print(div_term/2/math.pi)
        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0], x.shape[1], self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(0.1)
        self.min_max = (-2, +2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
        x_idxs = self.discretize(x)
        x_idxs += (
            torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        )
        # print(x_idxs,self.embeddings.weight.shape)
        return self.embeddings(x_idxs).mean(-2)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class SqueezeBetween0and1(nn.Module):  # take care of test set here
    def forward(self, x):
        width = x.max(0).values - x.min(0).values
        result = (x - x.min(0).values) / width
        result[(width == 0)[None].repeat(len(x), *[1] * (len(x.shape) - 1))] = 0.5
        return result


def get_normalized_uniform_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)), encoder_creator(in_dim, out_dim)
    )


def get_normalized_encoder(encoder_creator, data_std):
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.0, data_std), encoder_creator(in_dim, out_dim)
    )


def get_log_dims(x, eps=1e-10):
    logged_x = ((x + eps).log() - math.log(eps)) / (math.log(1.0 + eps) - math.log(eps))
    return logged_x


def add_log_neglog_dims(x, eps=1e-10):
    logged_x = get_log_dims(x, eps) / 2.0
    neglogged_x = 1 - get_log_dims(1 - x, eps) / 2.0
    logged_x[x > 0.5] = neglogged_x[x > 0.5]
    return torch.stack([x, logged_x], -1).view(*x.shape[:-1], -1)


class AddLogNegLogDims(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return add_log_neglog_dims(x, self.eps)


def get_logdim_encoder(encoder_creator, eps=1e-10):
    return lambda in_dim, out_dim: nn.Sequential(
        AddLogNegLogDims(eps), encoder_creator(in_dim * 2, out_dim)
    )


class ZNormalize(nn.Module):
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(-1, keepdim=True)) / std


class ZNormalizePerDataset(nn.Module):
    def forward(self, x):
        std = x.std(0, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(0, keepdim=True)) / std


class AppendEmbeddingEncoder(nn.Module):
    def __init__(self, base_encoder, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.base_encoder = base_encoder
        self.emb = nn.Parameter(torch.zeros(emsize))

    def forward(self, x):
        if (x[-1] == 1.0).all():
            append_embedding = True
        else:
            assert (x[-1] == 0.0).all(), (
                "You need to specify as last position whether to append embedding. "
                "If you don't want this behavior, please use the wrapped encoder instead."
            )
            append_embedding = False
        x = x[:-1]
        encoded_x = self.base_encoder(x)
        if append_embedding:
            encoded_x = torch.cat(
                [encoded_x, self.emb[None, None, :].repeat(1, encoded_x.shape[1], 1)], 0
            )
        return encoded_x


def get_append_embedding_encoder(encoder_creator):
    return lambda num_features, emsize: AppendEmbeddingEncoder(
        encoder_creator(num_features, emsize), num_features, emsize
    )


class VariableNumFeaturesEncoder(nn.Module):
    def __init__(self, base_encoder, num_features):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_features = num_features

    def forward(self, x):
        x = x * (self.num_features / x.shape[-1])
        x = torch.cat(
            (
                x,
                torch.zeros(
                    *x.shape[:-1], self.num_features - x.shape[-1], device=x.device
                ),
            ),
            -1,
        )
        return self.base_encoder(x)


def get_variable_num_features_encoder(encoder_creator):
    return lambda num_features, emsize: VariableNumFeaturesEncoder(
        encoder_creator(num_features, emsize), num_features
    )


class NoMeanEncoder(nn.Module):
    """
    This can be useful for any prior that is translation invariant in x or y.
    A standard GP for example is translation invariant in x.
    That is, GP(x_test+const,x_train+const,y_train) = GP(x_test,x_train,y_train).
    """

    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, x):
        return self.base_encoder(x - x.mean(0, keepdim=True))


def get_no_mean_encoder(encoder_creator):
    return lambda num_features, emsize: NoMeanEncoder(
        encoder_creator(num_features, emsize)
    )


MLP = lambda num_features, emsize: nn.Sequential(
    nn.Linear(num_features, emsize * 2), nn.ReLU(), nn.Linear(emsize * 2, emsize)
)


class NanHandlingEncoder(nn.Module):
    def __init__(self, num_features, emsize, keep_nans=True, layer_model=nn.Linear):
        super().__init__()
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.layer = layer_model(self.num_features, self.emsize)
        self.layer_model = layer_model

    def forward(self, x):
        if self.keep_nans:
            x = torch.cat(
                [
                    torch.nan_to_num(x, nan=0.0),
                    normalize_data(
                        torch.isnan(x) * -1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                    ),
                ],
                -1,
            )  # posinf and neginf should never occur in the data
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer(x)


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)


class StateActionEncoderCat(nn.Module):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.state_encoder = torch.nn.Sequential(nn.Linear(num_features - 3, 256), torch.nn.ReLU(),
                                                 nn.Linear(256, emsize // 4))
        self.action_encoder = torch.nn.Sequential(nn.Linear(3, 256), torch.nn.ReLU(),
                                             nn.Linear(256, emsize // 4))
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        state_enc = self.state_encoder(x[:, :, :11])
        action_enc = self.action_encoder(x[:, :, 11:])
        zero_pad = torch.full((x.shape[0], x.shape[1], self.emsize//2), 0., device="cuda:0")
        x = torch.cat((state_enc, action_enc, zero_pad), dim=2)
        return x

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)



class NextStateRewardEncoderCat(nn.Module):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.state_encoder = torch.nn.Sequential(nn.Linear(num_features - 3, 256), torch.nn.ReLU(),
                                                 nn.Linear(256, emsize // 4))
        self.r_encoder = torch.nn.Sequential(nn.Linear(1, 256), torch.nn.ReLU(),
                                             nn.Linear(256, emsize // 4))
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        state_enc = self.state_encoder(x[:, :, :11])
        r_enc = self.r_encoder(x[:, :, 13:])
        zero_pad = torch.full((x.shape[0], x.shape[1], self.emsize//2), 0., device="cuda:0")
        x = torch.cat((zero_pad, state_enc, r_enc), dim=2)
        return x

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)


class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)]
        )
        self.linear = nn.Linear(64, emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size * size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(
        self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs
    ):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(
        num_features, emsize, num_embs=num_embs_per_feature
    )


##### INPUT ENCODERS #####
class InputEncoder(nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize

    def forward(self, x, single_eval_pos=None):
        raise NotImplementedError


class LinearInputEncoder(InputEncoder):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.layer = nn.Linear(num_features, emsize)
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x, single_eval_pos=None):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return self.layer.forward(x)


class NanHandlingInputEncoderWrapper(InputEncoder):
    def __init__(self, num_features, emsize, base_encoder, keep_nans=True):
        super().__init__(num_features, emsize)
        self.num_features = 2 * num_features if keep_nans else num_features
        self.emsize = emsize
        self.keep_nans = keep_nans
        self.base_encoder = base_encoder(self.num_features, self.emsize)

    def forward(self, x, single_eval_pos=None):
        if self.keep_nans:
            x = torch.cat(
                [
                    torch.nan_to_num(x, nan=0.0),
                    normalize_data(
                        torch.isnan(x) * -1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 1
                        + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 2
                    ),
                ],
                -1,
            )
        else:
            x = torch.nan_to_num(x, nan=0.0)
        return self.base_encoder(x, single_eval_pos=single_eval_pos)


class DecomposedNumberInputEncoder(LinearInputEncoder):
    # Following https://arxiv.org/pdf/2112.01898.pdf
    def __init__(self, num_features, emsize, **args):
        embed_size_per_number = 3
        super().__init__(
            num_features=num_features * embed_size_per_number, emsize=emsize, **args
        )

    def forward(self, x, single_eval_pos=None):
        sign, mantissa, exponent = (
            torch.sign(torch.frexp(x).mantissa),
            torch.frexp(x).mantissa,
            torch.frexp(x).exponent,
        )
        x = torch.cat([sign, mantissa, exponent], -1)
        return super().forward(x)


class VariableNumFeaturesEncoderWrapper(InputEncoder):
    def __init__(
        self,
        num_features,
        emsize,
        base_encoder,
        normalize_by_used_features=True,
        remove_empty_features=False,
        columns_not_normalized=None,
    ):
        super().__init__(num_features, emsize)
        self.base_encoder = base_encoder(num_features, emsize)
        self.normalize_by_used_features = normalize_by_used_features
        self.remove_empty_features = remove_empty_features
        self.columns_not_normalized = columns_not_normalized

    def forward(self, x, single_eval_pos=None):
        sel = (x[1:] == x[0]).sum(0) != (
            x.shape[0] - 1
        )  # Indicator of empty features (Shape S, B, F)
        if self.remove_empty_features:
            for B in range(x.shape[1]):
                x[:, B, :] = torch.cat(
                    [
                        x[:, B, sel[B]],
                        torch.zeros(
                            x.shape[0], x.shape[-1] - sel[B].sum(), device=x.device
                        ),
                    ],
                    -1,
                )

        if self.normalize_by_used_features:
            x_old = x.clone()

            modified_sel = torch.clip(sel.sum(-1).unsqueeze(-1), min=1)
            x = x * (self.num_features / modified_sel)

            if self.columns_not_normalized is not None:
                x[:, :, self.columns_not_normalized] = x_old[
                    :, :, self.columns_not_normalized
                ]
        x = torch.cat(
            (
                x,
                torch.zeros(
                    *x.shape[:-1], self.num_features - x.shape[-1], device=x.device
                ),
            ),
            -1,
        )
        return self.base_encoder(x, single_eval_pos=single_eval_pos)


class InputNormalizationEncoderWrapper(InputEncoder):
    def __init__(
        self,
        num_features,
        emsize,
        base_encoder,
        normalize_on_train_only,
        normalize_to_ranking,
        normalize_x,
        remove_outliers,
        remove_unfilled_categoricals=False,
        columns_not_normalized=None,
    ):
        super().__init__(num_features, emsize)
        self.base_encoder = base_encoder(num_features, emsize)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_to_ranking = normalize_to_ranking
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.remove_unfilled_categoricals = remove_unfilled_categoricals
        self.columns_not_normalized = columns_not_normalized

    def forward(self, x: torch.Tensor, single_eval_pos=None):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        x_old = x.clone()

        if self.remove_unfilled_categoricals:
            # TODO: Use from utils import map_unique_to_order
            pass

        if self.normalize_to_ranking:
            x = to_ranking_low_mem(x)

        elif self.remove_outliers:
            x = remove_outliers(x, normalize_positions=normalize_position)

        if self.normalize_x:
            x = normalize_data(x, normalize_positions=normalize_position)

        if self.columns_not_normalized is not None:
            x[:, :, self.columns_not_normalized] = x_old[
                :, :, self.columns_not_normalized
            ]

        return self.base_encoder(x, single_eval_pos=single_eval_pos)

    # when loaded fix compatability issues
    def load_state_dict(self, state_dict, strict=True):
        if "columns_not_normalized" not in state_dict:
            state_dict["columns_not_normalized"] = None
        super().load_state_dict(state_dict, strict=strict)


##### TARGET ENCODERS #####
# TODO: Merge with InputEncoder
class TargetEncoder(nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize

    def forward(self, x, single_eval_pos=None):
        raise NotImplementedError

    def backward(self, y):
        raise NotImplementedError


# TODO: Instead of normalizing inputs to the transformer in training and for predictions separately
#  we could add normalization to the transformer encoder itself
class RegressionNormalizationEncoder(nn.Module):
    def __init__(self, num_features, emsize, base_encoder, normalize_on_train_only):
        super().__init__()
        self.num_features = num_features
        self.emsize = emsize
        self.base_encoder = base_encoder(num_features, emsize)
        self.normalize_on_train_only = normalize_on_train_only

    def forward(self, x, single_eval_pos=None):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        x, scaling = normalize_data(
            x, normalize_positions=normalize_position, return_scaling=True
        )
        return self.base_encoder(x, single_eval_pos=single_eval_pos), scaling

    def backward(self, y, scaling):
        raise NotImplementedError
