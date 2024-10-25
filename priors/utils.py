import traceback
import types
import inspect
import random
from functools import partial

import torch
import seaborn as sns
from typing import Callable, Optional

from utils import set_locals_in_self, normalize_data
from itertools import repeat
from .prior import Batch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import math

from torch.utils.data import IterableDataset, DataLoader


def make_dataloader(dataloader_kwargs, get_batch_kwargs, test_loader=False):
    if test_loader:
        get_batch_kwargs["batch_size"] = 1
        dataloader_kwargs["num_workers"] = 0
        dataloader_kwargs["pin_memory"] = False
        dataloader_kwargs.pop("prefetch_factor", None)  # Remove key if it exists
        dataloader_kwargs.pop("persistent_workers", None)  # Remove key if it exists

    ds = PriorDataset(**get_batch_kwargs)
    dl = DataLoader(
        ds,
        batch_sampler=None,
        batch_size=None,  # This disables automatic batching
        **dataloader_kwargs,
    )
    return dl


class PriorDataset(IterableDataset):
    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        eval_pos_seq_len_sampler: Callable,
        get_batch_method: Callable,
        num_features,
        seq_len_maximum: Optional[int] = None,
        device: Optional[str] = "cpu",
        test_loader: Optional[bool] = False,
        **get_batch_kwargs,
    ):
        # The stuff outside the or is set as class attribute before instantiation.
        self.num_features = num_features
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eval_pos_seq_len_sampler = eval_pos_seq_len_sampler
        self.seq_len_maximum = seq_len_maximum
        self.device = device
        self.get_batch_kwargs = get_batch_kwargs
        self.get_batch_method = get_batch_method
        self.model = None
        self.epoch = 0
        self.test_loader = test_loader
        print("DataLoader.__dict__", self.__dict__)

    def gbm(self):
        single_eval_pos, seq_len = self.eval_pos_seq_len_sampler()
        # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
        # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
        # if 'dynamic_batch_size' in kwargs and kwargs['dynamic_batch_size'] > 0 and kwargs[
        #    'dynamic_batch_size'] is not None:
        #    kwargs['batch_size'] = kwargs['batch_size'] * math.floor(
        #        math.pow(kwargs['seq_len_maximum'], kwargs['dynamic_batch_size'])
        #        / math.pow(kwargs['seq_len'], kwargs['dynamic_batch_size'])
        #    )
        batch = None
        exception = None
        for i in range(5):
            # Catch extremely rare errors and retry.
            try:
                batch: Batch = self.get_batch_method(
                    single_eval_pos=single_eval_pos,
                    seq_len=seq_len,
                    batch_size=self.batch_size,
                    num_features=self.num_features,
                    device=self.device,
                    model=self.model,
                    epoch=self.epoch,
                    test_batch=self.test_loader,
                    **self.get_batch_kwargs,
                )
                break
            except Exception as e:
                exception = e
                print("Exception in get_batch_method", e)
        if batch is None:
            traceback.print_tb(exception.__traceback__)
            raise Exception("Could not get batch after 5 retries.")

        if batch.single_eval_pos is None:
            batch.single_eval_pos = single_eval_pos

        return batch

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Different workers should have different seeds for numpy, python (pytorch automatic)
        num_steps = self.num_steps
        if worker_info is not None:
            np.random.seed(worker_info.seed % (2**32))
            random.seed(worker_info.seed % (2**32))
            num_steps = math.ceil(self.num_steps / worker_info.num_workers)
            num_steps = max(
                0, min(num_steps, self.num_steps - worker_info.id * num_steps)
            )

        # TODO: Why do we assign model, do we want to keep that behavior?
        # assert hasattr(self, 'model'), "Please assign model with `dl.model = ...` before training."
        return iter(self.gbm() for _ in range(num_steps))


def plot_features(
    data,
    targets,
    fig=None,
    categorical=True,
    plot_diagonal=True,
    append_y=False,
    categorical_indices=[],
    select_features_shuffled=False,
    N_features=4,
    n_samples=100,
):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

    if append_y:
        data = np.concatenate([data, np.expand_dims(targets, -1)], axis=-1)
    if select_features_shuffled:
        features_to_plot = np.random.choice(
            np.arange(0, data.shape[1]),
            size=min(data.shape[1], N_features),
            replace=False,
        )
    else:
        features_to_plot = np.arange(0, N_features)

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=N_features, nrows=N_features, figure=fig2)
    for d, d_ in enumerate(features_to_plot):
        for d2, d2_ in enumerate(features_to_plot):
            if d > d2:
                continue
            sub_ax = fig2.add_subplot(spec2[d, d2])
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if d == d2:
                if plot_diagonal:
                    if categorical:
                        sns.histplot(
                            data[:, d_],
                            multiple="stack",
                            hue=targets[:],
                            ax=sub_ax,
                            legend=False,
                            palette="deep",
                        )
                    else:
                        sns.histplot(
                            x=data[:, d_],
                            multiple="stack",
                            hue=np.digitize(
                                targets[:], np.histogram(targets[:], bins=20)[1]
                            ),
                            ax=sub_ax,
                            legend=False,
                            palette=sns.color_palette("rocket_r", as_cmap=True),
                        )
                sub_ax.set(ylabel=None)
            else:
                if categorical:
                    sns.scatterplot(
                        x=data[0:n_samples, d_],
                        y=data[0:n_samples, d2_],
                        hue=targets[0:n_samples],
                        legend=False,
                        palette="deep",
                    )
                else:
                    if d_ in categorical_indices or d2_ in categorical_indices:
                        g = sns.scatterplot(
                            y=data[0:n_samples, d2_],
                            x=data[0:n_samples, d_],
                            hue=targets[0:n_samples],
                            legend=False,
                        )
                        g.set(xticklabels=[])
                        g.set(xlabel=None)
                        g.tick_params(bottom=False)
                        g.set(yticklabels=[])
                        g.set(ylabel=None)
                    # elif d_ in categorical_indices and d2_ in categorical_indices:
                    #    sns.displot(y=data[0:n_samples, d2_],
                    #        x=data[0:n_samples, d_],)
                    else:
                        sns.scatterplot(
                            x=data[0:n_samples, d_],
                            y=data[0:n_samples, d2_],
                            hue=targets[0:n_samples],
                            legend=False,
                        )
                # plt.scatter(data[:, d], data[:, d2],
                #               c=targets[:])
            # sub_ax.get_xaxis().set_ticks([])
            # sub_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig2.show()


def plot_prior(prior, samples=1000, buckets=50):
    s = np.array([prior() for _ in range(0, samples)])
    count, bins, ignored = plt.hist(s, buckets, density=True)
    print(s.min())
    plt.show()


trunc_norm_sampler_f = lambda mu, sigma: lambda: stats.truncnorm(
    (0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma
).rvs(1)[0]
beta_sampler_f = lambda a, b: lambda: np.random.beta(a, b)
gamma_sampler_f = lambda a, b: lambda: np.random.gamma(a, b)
uniform_sampler_f = lambda a, b: lambda: np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b: lambda: round(np.random.uniform(a, b))


def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda: stats.rv_discrete(name="bounded_zipf", values=(x, weights)).rvs(1)


scaled_beta_sampler_f = lambda a, b, scale, minimum: lambda: minimum + round(
    beta_sampler_f(a, b)() * (scale - minimum)
)


def normalize_by_used_features_f(
    x, num_features_used, num_features, normalize_with_sqrt=False
):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features) ** (1 / 2)
    return x / (num_features_used / num_features)


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)  # .reshape(seq_len)
    x = x[
        order
    ]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y


def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


@torch.no_grad()
def sample_num_feaetures_get_batch(
    batch_size, seq_len, num_features, hyperparameters, get_batch, **kwargs
):
    if hyperparameters.get("sample_num_features", True) and not kwargs.get(
        "test_batch", False
    ):
        num_features = random.randint(1, num_features)
    return get_batch(
        batch_size, seq_len, num_features, hyperparameters=hyperparameters, **kwargs
    )


class CategoricalActivation(nn.Module):
    def __init__(
        self,
        categorical_p=0.1,
        ordered_p=0.7,
        keep_activation_size=False,
        num_classes_sampler=zipf_sampler_f(0.8, 1, 10),
    ):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = (
            torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None
        )

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros(
            (num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype
        )
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(
                dim=0
            ).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1], x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x


class QuantizationActivation(torch.nn.Module):
    def __init__(self, n_thresholds, reorder_p=0.5, normalize=True) -> None:
        super().__init__()
        self.n_thresholds = n_thresholds
        self.reorder_p = reorder_p
        self.thresholds = torch.nn.Parameter(torch.randn(self.n_thresholds))
        self.normalize = normalize
        self.quantize_percentiles = False

    def forward(self, x):
        if self.quantize_percentiles:
            # X is Seqlen, B
            threshold_indices = torch.randint(
                low=0, high=x.shape[0], size=(self.n_thresholds, x.shape[1])
            ).to(x.device)
            thresholds = torch.gather(x, 0, threshold_indices).detach()
            thresholds = torch.moveaxis(
                thresholds.transpose(0, -1)
                .unsqueeze(0)
                .repeat(x.shape[0], 1, 1)
                .unsqueeze(-2),
                -1,
                0,
            )
        else:
            mins, maxs = x.min(0)[0], x.max(0)[0]
            x = (x - mins) / (maxs - mins) - 0.5
            thresholds = torch.rand((self.n_thresholds, x.shape[1])) - 0.5
            thresholds = (
                thresholds.unsqueeze(1)
                .repeat(1, x.shape[0], 1)
                .unsqueeze(-1)
                .to(x.device)
            )
        # Threshold should be N_threshold, Seqlen, B, 1

        x = x.unsqueeze(-1)
        x_ = (x > thresholds).sum(0).squeeze(-1)

        if random.random() < self.reorder_p:
            x_ = randomize_classes(x_.unsqueeze(-1), self.n_thresholds).squeeze(-1)
        # x = ((x.float() - self.n_thresholds/2) / self.n_thresholds)# * data_std + data_mean
        if self.normalize:
            x_ = normalize_data(x_)
        return x_


def pretty_get_batch(get_batch):
    """
    Genereate string representation of get_batch function
    :param get_batch:
    :return:
    """
    if isinstance(get_batch, types.FunctionType):
        return f"<{get_batch.__module__}.{get_batch.__name__} {inspect.signature(get_batch)}"
    else:
        return repr(get_batch)


class get_batch_sequence(list):
    """
    This will call the get_batch_methods in order from the back and pass the previous as `get_batch` kwarg.
    For example for `get_batch_methods=[get_batch_1, get_batch_2, get_batch_3]` this will produce a call
    equivalent to `get_batch_3(*args,get_batch=partial(partial(get_batch_2),get_batch=get_batch_1,**kwargs))`.
    get_batch_methods: all priors, but the first, muste have a `get_batch` argument
    """

    def __init__(self, *get_batch_methods):
        if len(get_batch_methods) == 0:
            raise ValueError("Must have at least one get_batch method")
        super().__init__(get_batch_methods)

    def __repr__(self):
        s = ",\n\t".join([f"{pretty_get_batch(get_batch)}" for get_batch in self])
        return f"get_batch_sequence(\n\t{s}\n)"

    def __call__(self, *args, **kwargs):
        """

        Standard kwargs are: batch_size, seq_len, num_features
        This returns a priors.Batch object.
        """
        final_get_batch = self[0]
        for get_batch in self[1:]:
            final_get_batch = partial(get_batch, get_batch=final_get_batch)
        return final_get_batch(*args, **kwargs)
