from abc import ABCMeta, abstractmethod
from typing import Set, Optional, ClassVar
from dataclasses import dataclass, fields
import torch
from torch.utils.data import DataLoader
from functools import partial


def assert_no_nones(lis: list):
    assert all(
        e is not None for e in lis
    ), f"Merging attribute, where we don't know how to merge with Nones, {lis}"
    return lis


def triplet_tensor_merge_batches(attr, attr_name, batch_sizes, batch_dim=1):
    assert_no_nones(attr)
    assert all([a.shape[-1] == attr[0].shape[-1] for a in attr]), (
        f"Attr name: {attr_name} All tensors must have the same shape except for the first dimension."
        f" {attr_name} has shapes {[a.shape for a in attr]}"
    )
    return torch.cat(attr, batch_dim)


def list_merge(lists, f_name, batch_sizes):
    return sum(
        [
            ([None] * bs if sublist is None else sublist)
            for bs, sublist in zip(batch_sizes, lists)
        ],
        [],
    )


NOP = lambda *a: None


@dataclass
class Batch:
    """
    A batch of data, with non-optional x, y, and target_y attributes.
    All other attributes are optional.

    If you want to add an attribute for testing only, you can just assign it after creation like:
    ```
        batch = Batch(x=x, y=y, target_y=target_y)
        batch.test_attribute = test_attribute
    ```
    """

    ##################
    # Required entries
    ##################

    # The x inputs to the transformer
    x: torch.Tensor
    x_merge_func: ClassVar = triplet_tensor_merge_batches

    # The y input to the transformer, this is what the transformer is seeing as y values for the training data
    y: torch.Tensor
    y_merge_func: ClassVar = triplet_tensor_merge_batches

    # The target y values, this is what the transformer is trying to predict.
    # It is only used for the validation data during training.
    target_y: torch.Tensor
    target_y_merge_func: ClassVar = triplet_tensor_merge_batches

    ########################
    # Optional Batch Entries
    ########################

    # The style is given to the PFN to give hints for the kind of data it is seeing.
    style: Optional[torch.Tensor] = None
    style_merge_func: ClassVar = partial(triplet_tensor_merge_batches, batch_dim=0)

    # The style hyperparameter values are given to the PFN to give hints for the kind of data it is seeing.
    style_hyperparameter_values: Optional[torch.Tensor] = None

    # The single evaluation position is the cutoff between the training and validation data.
    single_eval_pos: Optional[int] = None
    single_eval_pos_merge_func: ClassVar = NOP

    # Gives info about the used dag, used for debugging mostly
    causal_model_dag: Optional[object] = None
    causal_model_dag_merge_func: ClassVar = list_merge

    # mean_prediction controls whether to do mean prediction in bar_distribution for nonmyopic BO
    mean_prediction: Optional[bool] = None
    mean_prediction_merge_func: ClassVar = NOP

    categorical_idxs: list = None
    categorical_idxs_merge_func: ClassVar = list_merge

    additional_x: torch.Tensor = None
    additional_x_merge_func: ClassVar = list_merge

    target_type: str = None
    target_type_merge_func: ClassVar = NOP

    censoring_time: torch.Tensor = None
    censoring_time_merge_func: ClassVar = triplet_tensor_merge_batches

    risks: torch.Tensor = None
    risks_merge_func: ClassVar = triplet_tensor_merge_batches

    event_times: torch.Tensor = None
    event_times_merge_func: ClassVar = triplet_tensor_merge_batches

    event_times_unobserved: torch.Tensor = None
    event_times_unobserved_merge_func: ClassVar = triplet_tensor_merge_batches

    event_observed: torch.Tensor = None
    event_observed_merge_func: ClassVar = triplet_tensor_merge_batches

    event_observed_masked_for_train: torch.Tensor = None
    event_observed_masked_for_train_merge_func: ClassVar = triplet_tensor_merge_batches

    def other_filled_attributes(
        self, set_of_attributes: Set[str] = frozenset(("x", "y", "target_y"))
    ):
        return [
            f.name
            for f in fields(self)
            if f.name not in set_of_attributes and getattr(self, f.name) is not None
        ]


def merge_batches(*batches, ignore_attributes=[]):
    """
    Merge all supported non-None fields in a pre-specified (general) way in batch dimesnsion,
    e.g. mutliple batch.x are concatenated in the batch dimension.
    :param ignore_attributes: attributes to remove from the merged batch, treated as if they were None.
    :return:
    """
    fields_to_be_merged = [
        f.name
        for f in fields(batches[0])
        if f.name not in ignore_attributes
        and any(getattr(b, f.name) is not None for b in batches)
    ]

    batch_sizes = [b.x.shape[1] for b in batches]

    # TODO: Check that fields does not return the merge funcs
    merge_funcs = {
        f.name: Batch.__dict__[f"{f.name}_merge_func"]
        for f in fields(batches[0])
        if f.name not in ignore_attributes
    }

    assert all(
        f in merge_funcs for f in fields_to_be_merged
    ), f"Unknown fields encountered in `safe_merge_batches_in_batch_dim`, {fields_to_be_merged}."
    return Batch(
        **{
            f: merge_funcs[f]([getattr(batch, f) for batch in batches], f, batch_sizes)
            for f in fields_to_be_merged
        }
    )
