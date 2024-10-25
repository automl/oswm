import contextlib
import itertools
import time
import warnings

import yaml
from contextlib import nullcontext
from tqdm import tqdm

import torch

import utils
import priors.prior
from transformer import TransformerModel

from utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    set_lr,
)
import positional_encodings
from utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from priors.utils import make_dataloader
import torch.multiprocessing as mp


class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")


def build_model(
    criterion,
    encoder_generator,
    test_batch,
    n_out,
    emsize=200,
    nhid=200,
    nlayers=6,
    seq_len=10,
    nhead=2,
    dropout=0.0,
    input_normalization=False,
    y_encoder_generator=None,
    decoder_dict={},
    extra_prior_kwargs_dict={},
    initializer=None,
    efficient_eval_masking=True,
    num_global_att_tokens=0,
    pos_encoder_generator=None,
    style_encoder_generator=None,
    **model_extra_args,
):
    decoder_dict = decoder_dict if decoder_dict else {"standard": (None, n_out)}

    decoder_once_dict = {}

    # For survival models an extra column is supported to encode the censoring variable
    survival_extra_column = 0

    if style_encoder_generator is not None:
        style_def = test_batch.style

        print(
            f"Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}"
        )
        style_encoder = (
            style_encoder_generator(style_def.shape[1], emsize)
            if (style_def is not None)
            else None
        )
    else:
        style_encoder = None

    pos_encoder = (pos_encoder_generator or positional_encodings.NoPositionalEncoding)(
        emsize, seq_len * 2
    )

    encoder = encoder_generator(
        extra_prior_kwargs_dict["num_features"] + survival_extra_column, emsize
    )
    model = TransformerModel(
        encoder=encoder,
        nhead=nhead,
        ninp=emsize,
        nhid=nhid,
        nlayers=nlayers,
        dropout=dropout,
        style_encoder=style_encoder,
        y_encoder=y_encoder_generator(14, emsize),  # TODO: Num outputs = y_def.shape?
        input_normalization=input_normalization,
        pos_encoder=pos_encoder,
        decoder_dict=decoder_dict if decoder_dict else {"standard": (None, n_out)},
        init_method=initializer,
        efficient_eval_masking=efficient_eval_masking,
        decoder_once_dict=decoder_once_dict,
        num_global_att_tokens=num_global_att_tokens,
        **model_extra_args,
    )

    return model


# TODO: Split train in train and build model
# TODO: Use PytorchLightning?
def train(
    get_batch_method,
    criterion,
    encoder_generator,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    seq_len=10,
    bptt=None,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    decoder_dict={},
    extra_prior_kwargs_dict={},
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    validation_period=10,
    single_eval_pos_gen=None,
    bptt_extra_samples=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    epoch_callback=None,
    step_callback=None,
    continue_model=None,
    initialize_with_model=None,
    train_mixed_precision=False,
    progress_bar=True,
    num_classes=0,
    dataloader_kwargs={},
    dataloader_device=None,
    optimizer_state=None,
    scaler_state=None,
    trained_epochs_until_now=0,
    **model_extra_args,
):
    # TODO: Check wheather to use pytorch lightning and make configuration simpler
    if bptt_extra_samples is not None:
        warnings.warn(
            "bptt_extra_samples is not supported right now, someone needs to implement it."
        )
    if bptt is not None:
        warnings.warn("We renamed `bptt` to `seq_len`. Please use `seq_len` instead.")

    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )
    dataloader_device = device if dataloader_device is None else dataloader_device

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples and False:  # TODO: Currently disabled
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, seq_len

    get_batch_kwargs = {
        "batch_size": batch_size,
        "get_batch_method": get_batch_method,
        "num_steps": steps_per_epoch,
        "eval_pos_seq_len_sampler": eval_pos_seq_len_sampler,
        "device": dataloader_device if dataloader_device is not None else device,
        "epoch_count": trained_epochs_until_now,
        **extra_prior_kwargs_dict,
    }
    if "cuda" in dataloader_device and dataloader_kwargs.get("num_workers", 0) > 0:
        mp.set_start_method("spawn", force=True)  # fork does not work with cuda
    dl = make_dataloader(
        get_batch_kwargs=get_batch_kwargs, dataloader_kwargs=dataloader_kwargs
    )

    print(
        f"Initialized data loader with {steps_per_epoch} steps and {batch_size} batch size"
    )

    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    # We use a string-based comparison in addition to a class-based comparison
    # We do this because `reload` changes class hierarchies, making `isinstance` calls wrong when editing the classes
    elif ("BarDistribution" in criterion.__class__.__name__):
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = num_classes
    elif isinstance(criterion, nn.MSELoss):
        n_out = 14
    else:
        n_out = 1

    dl_test = make_dataloader(
        get_batch_kwargs=get_batch_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        test_loader=True,
    )
    test_batch: priors.Batch = next(iter(dl_test))

    if continue_model:
        model = continue_model
    else:
        # TODO: Update train rewrite to address the issue of having both continue_model and load state dict.
        model = build_model(
            criterion,
            encoder_generator,
            test_batch,
            n_out,
            decoder_dict=decoder_dict,
            extra_prior_kwargs_dict=extra_prior_kwargs_dict,
            **model_extra_args,
        )

    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    # TODO: Check if this is still needed, remove otherwise (I (Noah) don't think it is needed)
    try:
        for (k, v), (k2, v2) in zip(
            model.state_dict().items(), initialize_with_model.state_dict().items()
        ):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=test_batch.mean_prediction is not None,
        )
        dl.dataset.model = (
            model.module
        )  # use local model, should not use multi-gpu functionality..
    else:
        dl.dataset.model = model

    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # the lower bound max(epochs,1) is to avoid division by zero when only loading a model and not training
    scheduler = scheduler(warmup_share=warmup_epochs / max(epochs, 1), max_lr=lr)

    scaler = GradScaler() if train_mixed_precision and utils.is_cuda(device) else None

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if scaler_state is not None and scaler is not None:
        scaler.load_state_dict(scaler_state)

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        total_positional_losses = 0.0
        total_positional_losses_recorded = 0
        time_to_get_batch = 0.0
        forward_time = 0.0
        step_time = 0.0
        nan_steps = torch.tensor(0.0, device=device)
        ignore_steps = torch.tensor(0.0, device=device)
        before_get_batch = time.time()

        def get_metrics():
            def robust_divide(a, b, list=False):
                try:
                    if list:
                        return (a / b).tolist()
                    return a / b
                except ZeroDivisionError:
                    return 0

            return (
                robust_divide(total_loss, steps_per_epoch),
                robust_divide(
                    total_positional_losses, total_positional_losses_recorded, list=True
                ),
                time_to_get_batch,
                forward_time,
                step_time,
                nan_steps.cpu().item() / (batch + 1),
                ignore_steps.cpu().item() / (batch + 1),
            )

        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        tqdm_iter = tqdm(
            dl, desc="Training Epoch", disable=not (rank == 0 and progress_bar)
        )  # , disable=not verbose

        for batch, full_data in enumerate(tqdm_iter):
            # Get batch inputs
            data, targets, single_eval_pos = (
                (
                    full_data.style.to(device)
                    if full_data.style is not None
                    else full_data.style,
                    full_data.x.to(device),
                    full_data.y.to(device),
                ),
                full_data.y.to(device),
                full_data.single_eval_pos,
            )

            if using_dist and not (
                batch % aggregate_k_gradients == aggregate_k_gradients - 1
            ):
                cm = model.no_sync()
            else:
                cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()
                # TODO: This disables the bptt_extra_samples functionality but otherwise single eval pos is overwritten
                # if bptt_extra_samples is None:
                #    single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                # else:
                #    single_eval_pos = targets.shape[0] - bptt_extra_samples
                metrics_to_log = {}
                # Measure Nan share before losses disable nans internally
                nan_share_through_targets = float(
                    targets[torch.isnan(targets)].numel()
                ) / float(targets.numel())
                with autocast() if scaler is not None else contextlib.nullcontext():
                    # If style is set to None, it should not be transferred to device
                    out = model(
                        tuple(e.to(device) if torch.is_tensor(e) else e for e in data),
                        single_eval_pos=single_eval_pos,
                        only_return_standard_out=False,
                    )

                    # this handling is for training old models only, this can be deleted soon(ish)
                    # to only support models that return a tuple of dicts
                    out, output_once = out if isinstance(out, tuple) else (out, None)
                    output = out["standard"]

                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]

                    if len(targets.shape) == len(output.shape):
                        # this implies the prior uses a trailing 1 dimesnion
                        # below we assume this not to be the case
                        targets = targets.squeeze(-1)
                    assert targets.shape[:1] == output.shape[:1], (
                        f"Target shape {targets.shape[:1]} "
                        "does not match output shape {output.shape[:1]}"
                    )
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert (
                            output.shape[-1] == 2
                        ), "need to write a little bit of code to handle multiple regression targets at once"

                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(
                            mean_pred.flatten(),
                            targets.flatten(),
                            var=var_pred.flatten(),
                        )
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        targets[torch.isnan(targets)] = -100
                        losses = criterion(output.flatten(), targets.flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        targets[torch.isnan(targets)] = -100
                        if targets.max() >= output.shape[-1]:
                            targets[:] = -100
                            print(
                                f"Invalid target value ({targets.max()}), skipping batch..."
                            )
                        losses = criterion(
                            output.reshape(-1, n_out), targets.long().flatten()
                        )
                    elif extra_prior_kwargs_dict.get(
                        "use_censoring_loss", False
                    ) and extra_prior_kwargs_dict.get("is_survival", False):
                        # Nan Targets are removed from the loss (Loss is set to 0) which changes loss scaling

                        losses, censoring_losses, time_losses = criterion(
                            output.to(device),
                            time=full_data.event_times[single_eval_pos:].to(device),
                            event=full_data.event_observed[single_eval_pos:].to(device),
                        )
                        metrics_to_log.update(
                            {
                                "censoring_loss": float(
                                    utils.torch_nanmean(
                                        censoring_losses.mean(0),
                                        return_nanshare=False,
                                    )
                                ),
                                "time_loss": float(
                                    utils.torch_nanmean(
                                        time_losses.mean(0), return_nanshare=False
                                    )
                                ),
                            }
                        )
                    elif isinstance(criterion):
                        # Nan Targets are removed from the loss (Loss is set to 0) which changes loss scaling
                        losses = criterion(output, targets)
                    else:
                        raise NotImplementedError(f"{criterion} not implemented")
                    # we reshape with -1 first, to allow appending the loss for the mean prediction
                    # this will be made into losses = losses.view(output.shape[:2]) when we remove nonmyopic code
                    losses = losses.view(-1, output.shape[1])
                    loss, nan_share_through_losses = utils.torch_nanmean(
                        losses.mean(0), return_nanshare=True
                    )

                    nan_share_through_targets = (
                        nan_share_through_losses + nan_share_through_targets
                    )
                    loss_scaled = loss / aggregate_k_gradients

                if scaler:
                    loss_scaled = scaler.scale(loss_scaled)

                loss_scaled.backward()

                set_lr(
                    optimizer, scheduler(((epoch + 1) + batch / len(dl)) / epochs)
                )  # because we start at 1
                if not torch.isnan(loss_scaled):
                    if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    total_loss += loss.cpu().detach().item()
                    total_positional_losses += (
                        losses.mean(1).cpu().detach()
                        if single_eval_pos is None
                        else nn.functional.one_hot(
                            torch.tensor(single_eval_pos), seq_len
                        )
                        * utils.torch_nanmean(
                            losses[: seq_len - single_eval_pos].mean(0)
                        )
                        .cpu()
                        .detach()
                    )

                    total_positional_losses_recorded += (
                        torch.ones(seq_len)
                        if single_eval_pos is None
                        else nn.functional.one_hot(
                            torch.tensor(single_eval_pos), seq_len
                        )
                    )

                    metrics_to_log = {
                        **metrics_to_log,
                        **{
                            f"loss": float(loss),
                            "single_eval_pos": float(single_eval_pos),
                        },
                    }
                    if step_callback is not None and rank == 0:
                        step_callback(metrics_to_log, step=batch)
                    nan_steps += nan_share_through_targets
                    ignore_steps += (targets == -100).float().mean()
                else:
                    print(
                        f"All batches have NaN in loss. Target NaN {torch.isnan(targets).any()}"
                        f" output NaN {torch.isnan(output).any()}"
                    )
                    continue
            tqdm_iter.set_postfix(
                {
                    "data_time": time_to_get_batch,
                    "step_time": step_time,
                    "mean_loss": total_loss / (batch + 1),
                    **metrics_to_log,
                },
                refresh=False,
            )
            before_get_batch = time.time()
        return get_metrics()

    total_loss = 0.0
    total_positional_losses = [float("inf")]
    try:
        # Initially test the epoch callback function
        if epoch_callback is not None and rank == 0:
            epoch_callback(
                _unwrap_model_dl(model)[0],
                1 if continue_model is None else trained_epochs_until_now + 1,
                data_loader=dl,
                optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict() if scaler is not None else None,
                extra_infos={"lr": optimizer.param_groups[0]["lr"]},
            )
        for epoch in (
            range(trained_epochs_until_now + 1, epochs + 1)
            if epochs is not None
            else itertools.count(1)
        ):
            epoch_start_time = time.time()
            try:
                dl.dataset.epoch = epoch
                (
                    total_loss,
                    total_positional_losses,
                    time_to_get_batch,
                    forward_time,
                    step_time,
                    nan_share,
                    ignore_share,
                ) = train_epoch()
            except Exception as e:
                print(f"Invalid epoch encountered, skipping {e}...")
                continue
            if hasattr(dl, "validate") and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print("-" * 89)
                print(
                    f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | "
                    f"pos losses {','.join([('-' if l != l else f'{l:5.2f}') for l in total_positional_losses])}, lr of zeroth group {optimizer.param_groups[0]['lr']}"
                    f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
                    f" forward time {forward_time:5.2f}"
                    f" nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}"
                    + (f"val score {val_score}" if val_score is not None else "")
                )
                print("-" * 89)

            if epoch_callback is not None and rank == 0:
                epoch_callback(
                    _unwrap_model_dl(model)[0],
                    epoch,
                    data_loader=dl,
                    optimizer_state=optimizer.state_dict(),
                    scaler_state=scaler.state_dict() if scaler is not None else None,
                    extra_infos={"lr": optimizer.param_groups[0]["lr"]},
                )
    except KeyboardInterrupt:
        pass

    if rank == 0:  # trivially true for non-parallel training
        model, dl = _unwrap_model_dl(model, dl)
        # This is needed to allow pickling of results.
        # It does not change any functionality, but only makes the next epoch a bit slower.
        if hasattr(dl, "_iterator"):
            dl._iterator = None
        return total_loss, total_positional_losses, model.to("cpu"), dl


def _unwrap_model_dl(model, dl=None):
    """
    Unwraps the encapsulated model in case distributed training is used. In
    that case also the returned dataloader is set to None. Otherwise, the
    function is a no-op.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module, None

    return model, dl


def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
