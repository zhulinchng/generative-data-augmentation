"""
Helper script for training and evaluation of models.
This script contains helper functions for distributed training, model setup, evaluation, and more.

Most of the training and evaluation code are modified from the TorchVision repository, to ensure models are trained with the same methods for comparison purposes.

References:
TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
"""

import copy
import datetime
import hashlib
import math
import os
import time
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from typing import List, Optional, Tuple

import schedulefree
import torch
import torch.distributed as dist
import torch.utils.data
from torch import Tensor, nn
from torchvision.transforms import functional as F
from tqdm import tqdm

import wandb


def setup_for_distributed(is_master):
    """
    This function disables printing when not in the master process.

    Parameters:
    - is_master (bool): A boolean value indicating whether the current process is the master process.

    Returns:
    - None

    This function replaces the built-in print function with a custom print function that only prints when the current process is the master process. It allows for an optional 'force' keyword argument to override this behavior and force printing even when not in the master process.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(setup=dict()):
    """
    Initializes the distributed mode for training.

    Args:
        setup (dict): A dictionary containing setup parameters.

    Returns:
        dict: A dictionary containing the updated setup parameters.

    Raises:
        None

    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        setup["rank"] = int(os.environ["RANK"])
        setup["world_size"] = int(os.environ["WORLD_SIZE"])
        setup["gpu"] = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        setup["rank"] = int(os.environ["SLURM_PROCID"])
        setup["gpu"] = setup["rank"] % torch.cuda.device_count()
    elif setup.get("rank") is not None:
        pass
    else:
        print("Not using distributed mode")
        setup["distributed"] = False
        return setup

    setup["distributed"] = True

    torch.cuda.set_device(setup["gpu"])
    setup["dist_backend"] = "nccl"
    print(f"| distributed init (rank {setup['rank']}): {setup['dist_url']}", flush=True)
    torch.distributed.init_process_group(
        backend=setup["dist_backend"],
        init_method=setup["dist_url"],
        world_size=setup["world_size"],
        rank=setup["rank"],
    )
    torch.distributed.barrier()
    setup_for_distributed(setup["rank"] == 0)
    return setup


def reduce_across_processes(val):
    """
    Reduces the input value across all processes using the `torch.distributed` package.

    Args:
        val: The value to be reduced.

    Returns:
        The reduced value.

    Raises:
        None
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights. Original implementation taken from:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
                weights_only=True,
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}"
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            checkpoint[checkpoint_key], "module."
        )
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    """
    Sets weight decay for different parameter groups in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        weight_decay (float): The weight decay value for parameters not falling into any other group.
        norm_weight_decay (float, optional): The weight decay value for parameters belonging to normalization layers.
        norm_classes (List[type], optional): A list of normalization layer classes.
        custom_keys_weight_decay (List[Tuple[str, float]], optional): A list of custom key-value pairs for weight decay.

    Returns:
        List[Dict[str, Any]]: A list of parameter groups with their respective weight decay values.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = (
                    f"{prefix}.{name}" if prefix != "" and "." in key else name
                )
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    num_classes = len(data_loader.dataset.classes)
    if num_classes < 5:
        print(f"Top K is not meaningful for few classes. Setting topk to {num_classes}")
        topk = (1, num_classes)
    else:
        topk = (1, 5)
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=topk)
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    metrics = dict()
    metrics["acc1"] = metric_logger.acc1.global_avg
    metrics["acc5"] = metric_logger.acc5.global_avg
    metrics["loss"] = metric_logger.loss.global_avg
    metrics["count"] = metric_logger.loss.count
    return metrics


class RandomMixUp(torch.nn.Module):
    """Randomly apply MixUp to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutMix(torch.nn.Module):
    """Randomly apply CutMix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(
                "Please provide a valid positive value for the num_classes."
            )
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=self.num_classes
            ).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


def model_setup(model, setup: dict):
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    model_without_ddp = model
    if setup["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[setup["gpu"]]
        )
        model_without_ddp = model.module

    model_ema = None
    if setup["model_ema"]:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = (
            setup["world_size"]
            * setup["batch_size"]
            * setup["model_ema_steps"]
            / setup["epochs"]
        )
        alpha = 1.0 - setup["model_ema_decay"]
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(
            model_without_ddp, device=torch.device(setup["device"]), decay=1.0 - alpha
        )
    return model, model_ema, model_without_ddp


def getValSampler(val_dataset, setup: dict):
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if setup["distributed"]:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return val_sampler


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def getTrainSampler(train_dataset, setup: dict):
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    if setup["distributed"]:
        if setup["ra_sampler"]:
            train_sampler = RASampler(
                train_dataset, shuffle=True, repetitions=setup["ra_reps"]
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    return train_sampler


def trainSetup(
    model,
    setup: dict,
    val_dataloader: torch.utils.data.dataloader.DataLoader,
    device: torch.device,
):
    """
    Sets up the training configuration for the model.

    Args:
        model: The model to be trained.
        setup (dict): A dictionary containing the setup configuration.
        val_dataloader (torch.utils.data.dataloader.DataLoader): The validation dataloader.
        device (torch.device): The device to be used for training.

    Returns:
        Tuple: A tuple containing the following elements:
            - model: The model to be trained.
            - criterion: The loss criterion.
            - optimizer: The optimizer.
            - setup (dict): The setup configuration.
            - model_ema: The exponential moving average model.
            - scaler: The gradient scaler for mixed precision training.
            - model_without_ddp: The model without distributed data parallel.
            - lr_scheduler: The learning rate scheduler.
            - run: The W&B run object.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    criterion = nn.CrossEntropyLoss(label_smoothing=setup["label_smoothing"])
    custom_keys_weight_decay = []
    if setup["bias_weight_decay"] is not None:
        custom_keys_weight_decay.append(("bias", setup["bias_weight_decay"]))
    if setup["transformer_embedding_decay"] is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, setup["transformer_embedding_decay"]))
    parameters = set_weight_decay(
        model,
        setup["weight_decay"],
        norm_weight_decay=setup["norm_weight_decay"],
        custom_keys_weight_decay=(
            custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None
        ),
    )
    opt_name = setup["opt"].lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=setup["lr"],
            momentum=setup["momentum"],
            weight_decay=setup["weight_decay"],
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=setup["lr"],
            momentum=setup["momentum"],
            weight_decay=setup["weight_decay"],
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=setup["lr"], weight_decay=setup["weight_decay"]
        )
    elif opt_name == "schedulefree_sgd":
        optimizer = schedulefree.SGDScheduleFree(
            parameters,
            lr=setup["lr"],
            momentum=setup["momentum"],
            weight_decay=setup["weight_decay"],
        )
    elif opt_name == "schedulefree_adamw":
        optimizer = schedulefree.AdamWScheduleFree(
            parameters, lr=setup["lr"], weight_decay=setup["weight_decay"]
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {setup['opt']}. Only SGD, RMSprop and AdamW are supported."
        )
    scaler = torch.cuda.amp.GradScaler() if setup["amp"] else None
    setup["lr_scheduler"] = setup["lr_scheduler"].lower()
    schedulefree_opt = False
    if opt_name.startswith("schedulefree_"):
        lr_scheduler = None
        schedulefree_opt = True
    elif setup["lr_scheduler"] == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=setup["lr_step_size"], gamma=setup["lr_gamma"]
        )
    elif setup["lr_scheduler"] == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=setup["epochs"] - setup["lr_warmup_epochs"],
            eta_min=setup["lr_min"],
        )
    elif setup["lr_scheduler"] == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=setup["lr_gamma"]
        )
    elif setup["lr_scheduler"] == "multisteplr":
        if setup["lr_milestones"] is None:
            raise RuntimeError(
                "lr_milestones must be provided for MultiStepLR scheduler."
            )
        elif len(setup["lr_milestones"]) == 0:
            warnings.warn(
                "Empty lr_milestones provided for MultiStepLR scheduler. "
                "No learning rate changes will be applied."
            )
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=setup["lr_milestones"], gamma=setup["lr_gamma"]
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{setup['lr_scheduler']}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
    if setup["lr_warmup_epochs"] > 0 and not schedulefree_opt:
        if setup["lr_warmup_method"] == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=setup["lr_warmup_decay"],
                total_iters=setup["lr_warmup_epochs"],
            )
        elif setup["lr_warmup_method"] == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=setup["lr_warmup_decay"],
                total_iters=setup["lr_warmup_epochs"],
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{setup['lr_warmup_method']}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[setup["lr_warmup_epochs"]],
        )
    elif schedulefree_opt:
        print("Using ScheduleFree optimizer, no lr scheduler will be used.")
    else:
        lr_scheduler = main_lr_scheduler
    model_without_ddp = model
    if setup["distributed"]:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[setup["gpu"]]
        )
        model_without_ddp = model.module
    model_ema = None
    if setup["model_ema"]:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = (
            setup["world_size"]
            * setup["batch_size"]
            * setup["model_ema_steps"]
            / setup["epochs"]
        )
        alpha = 1.0 - setup["model_ema_decay"]
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )
    if setup["resume"]:
        wandb_id = setup["wandb_id"]
        checkpoint = torch.load(setup["resume"], map_location="cpu", weights_only=False)
        if checkpoint["setup"]["distributed"]:
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                checkpoint["model"], "module."
            )
        model_without_ddp.load_state_dict(checkpoint["model"])
        # model = model_without_ddp
        if not setup["test_only"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if not schedulefree_opt:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if setup["lr_scheduler"] == "multisteplr":
                # update the milestones to only include future epochs
                setup["lr_milestones"] = [
                    milestone
                    for milestone in setup["lr_milestones"]
                    if milestone >= checkpoint["epoch"]
                ]
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=setup["lr_milestones"],
                    gamma=setup["lr_gamma"],
                )
                for _ in range(checkpoint["epoch"] + 1):
                    lr_scheduler.step()
        setup["start_epoch"] = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        wandb_id = wandb.util.generate_id()
        print(f"wandb_id: {wandb_id}")
    if setup["test_only"]:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(
                model_ema, criterion, val_dataloader, device=device, log_suffix="EMA"
            )
        else:
            evaluate(model, criterion, val_dataloader, device=device)
    if setup["distributed"]:
        run = wandb.init(
            id=wandb_id,
            entity=setup["entity"],
            project=setup["project"],
            group="DDP",
            job_type="training",
            config=setup,
            mode="offline",
            resume="allow",
        )
    else:
        run = wandb.init(
            id=wandb_id,
            entity=setup["entity"],
            project=setup["project"],
            job_type="training",
            config=setup,
            mode="offline",
            resume="allow",
        )
        run.watch(model)

    return (
        model,
        criterion,
        optimizer,
        setup,
        model_ema,
        scaler,
        model_without_ddp,
        lr_scheduler,
        run,
    )


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    setup,
    model_ema=None,
    scaler=None,
    run=None,
):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        data_loader (torch.utils.data.DataLoader): The data loader for loading training data.
        device (torch.device): The device to be used for training.
        epoch (int): The current epoch number.
        setup (dict): A dictionary containing setup configurations.
        model_ema (torch.nn.Module, optional): An exponential moving average model. Defaults to None.
        scaler (torch.cuda.amp.GradScaler, optional): A gradient scaler for mixed precision training. Defaults to None.
        run (wandb.Run, optional): A Weights & Biases run object for logging. Defaults to None.

    Returns:
        tuple: A tuple containing the updated model, optimizer, model_ema, scaler, metric_logger, and run objects.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    model.train()
    schedulefree_opt = setup["opt"].lower().startswith("schedulefree_")
    if schedulefree_opt:
        optimizer.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value}"))
    do_log = run is not None
    wandb_loss, wandb_acc1, wandb_acc5, wandb_throughput = 0, 0, 0, 0
    n_batches = len(data_loader)
    num_classes = len(data_loader.dataset.classes)
    if num_classes < 5:
        topk = (1, num_classes)
    else:
        topk = (1, 5)
    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, setup["print_freq"], header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if setup["clip_grad_norm"] is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), setup["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if setup["clip_grad_norm"] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), setup["clip_grad_norm"])
            optimizer.step()

        if model_ema and i % setup["model_ema_steps"] == 0:
            model_ema.update_parameters(model)
            if epoch < setup["lr_warmup_epochs"] and not schedulefree_opt:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = accuracy(output, target, topk=topk)
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            throughput := batch_size / (time.time() - start_time)
        )
        wandb_loss += loss.item()
        wandb_acc1 += acc1.item()
        wandb_acc5 += acc5.item()
        wandb_throughput += throughput
        if do_log and i % setup["print_freq"] == 0 and i != 0:
            run.log(
                {
                    "batch/batch": i,
                    "batch/train_loss": wandb_loss / setup["print_freq"],
                    "batch/train_acc1": wandb_acc1 / setup["print_freq"],
                    "batch/train_acc5": wandb_acc5 / setup["print_freq"],
                    "batch/train_img_s": wandb_acc5 / setup["print_freq"],
                }
            )
            wandb_loss, wandb_acc1, wandb_acc5, wandb_throughput = 0, 0, 0, 0
    remainder = n_batches - n_batches // setup["print_freq"] * setup["print_freq"]
    if do_log and remainder > 0 and n_batches > setup["print_freq"]:
        run.log(
            {
                "batch/batch": i,
                "batch/train_loss": wandb_loss / remainder,
                "batch/train_acc1": wandb_acc1 / remainder,
                "batch/train_acc5": wandb_acc5 / remainder,
                "batch/train_img_s": wandb_acc5 / remainder,
            }
        )
    return model, optimizer, model_ema, scaler, metric_logger, run


def train(
    model,
    model_without_ddp,
    criterion,
    optimizer,
    lr_scheduler,
    train_loader,
    train_sampler,
    val_loader,
    device,
    setup: dict,
    model_ema=None,
    scaler=None,
    run=None,
):
    """
    Trains the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        model_without_ddp (nn.Module): The model without DistributedDataParallel wrapper.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer for updating model parameters.
        lr_scheduler (LRScheduler): The learning rate scheduler.
        train_loader (DataLoader): The data loader for training data.
        train_sampler (Sampler): The sampler for training data.
        val_loader (DataLoader): The data loader for validation data.
        device (torch.device): The device to be used for training.
        setup (dict): A dictionary containing setup parameters.
        model_ema (nn.Module, optional): The exponential moving average model. Defaults to None.
        scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler for mixed precision training. Defaults to None.
        run (comet_ml.Experiment, optional): The experiment object for logging. Defaults to None.

    Returns:
        nn.Module: The trained model.
        run: The W&B run object.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    print("Start training")
    start_time = time.time()
    do_log = run is not None
    schedulefree_opt = setup["opt"].lower().startswith("schedulefree_")
    highest_test_acc1 = 0
    highest_test_acc5 = 0
    lowest_test_loss = float("inf")
    for epoch in tqdm(range(setup["start_epoch"], setup["epochs"])):
        if setup["distributed"]:
            train_sampler.set_epoch(epoch)

        if not schedulefree_opt:
            epoch_lr = lr_scheduler.get_last_lr()[0]
        else:
            epoch_lr = optimizer.param_groups[0]["lr"]

        model, optimizer, model_ema, scaler, metric_logger, run = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch,
            setup,
            model_ema,
            scaler,
            run,
        )

        if not schedulefree_opt:
            lr_scheduler.step()
        else:
            optimizer.eval()

        if model_ema:
            val_metrics = evaluate(
                model_ema,
                criterion,
                val_loader,
                device=device,
                log_suffix="EMA",
                print_freq=setup["print_freq"],
            )
        else:
            val_metrics = evaluate(
                model,
                criterion,
                val_loader,
                device=device,
                print_freq=setup["print_freq"],
            )
        if setup["output_dir"]:
            if val_metrics["acc1"] > highest_test_acc1:
                highest_test_acc1 = val_metrics["acc1"]
                highest_test_acc5 = val_metrics["acc5"]
                lowest_test_loss = val_metrics["loss"]
                save_interval = True
                print(
                    f"New highest test acc1: {highest_test_acc1:.3f} at epoch {epoch}."
                )
            elif (
                val_metrics["acc1"] == highest_test_acc1
                and val_metrics["acc5"] > highest_test_acc5
            ):
                highest_test_acc5 = val_metrics["acc5"]
                lowest_test_loss = val_metrics["loss"]
                save_interval = True
                print(
                    f"New highest test acc5: {highest_test_acc5:.3f} at epoch {epoch}."
                )
            elif (
                val_metrics["acc1"] == highest_test_acc1
                and val_metrics["acc5"] == highest_test_acc5
                and val_metrics["loss"] < lowest_test_loss
            ):
                lowest_test_loss = val_metrics["loss"]
                save_interval = True
                print(f"New lowest test loss: {lowest_test_loss:.3f} at epoch {epoch}.")
            elif epoch < 100 and (epoch % 10 == 0 or setup["epochs"] - 1 == epoch):
                save_interval = True
            elif (
                epoch < 1000
                and epoch > 100
                and (epoch % 100 == 0 or setup["epochs"] - 1 == epoch)
            ):
                save_interval = True
            elif epoch > 1000 and (epoch % 1000 == 0 or setup["epochs"] - 1 == epoch):
                save_interval = True
            else:
                save_interval = False
            if save_interval:
                model_without_ddp = model
                if not schedulefree_opt:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "setup": setup,
                    }
                else:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "setup": setup,
                    }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                save_on_master(
                    checkpoint, os.path.join(setup["output_dir"], f"model_{epoch}.pth")
                )
                save_on_master(
                    checkpoint, os.path.join(setup["output_dir"], "checkpoint.pth")
                )
        if do_log:
            run.log(
                {
                    "epoch/epoch": epoch,
                    "epoch/train_learning_rate": epoch_lr,
                    "epoch/train_acc1": metric_logger.meters["acc1"].global_avg,
                    "epoch/train_acc5": metric_logger.meters["acc5"].global_avg,
                    "epoch/train_loss": metric_logger.meters["loss"].global_avg,
                    "epoch/train_img_s": metric_logger.meters["img/s"].global_avg,
                    "epoch/test_acc1": val_metrics["acc1"],
                    "epoch/test_acc5": val_metrics["acc5"],
                    "epoch/test_loss": val_metrics["loss"],
                }
            )
        print(f"Epoch: {epoch} completed.")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    return model, run


def evaluateByClass(
    model,
    data_loader,
    num_classes,
    device,
    top_n=5,
    print_freq=100,
    log_suffix="",
    actual_target=None,
):
    """
    Evaluate the model's performance on a given dataset by class.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): The data loader for the dataset.
        num_classes (int): The number of classes in the dataset.
        device (torch.device): The device to perform the evaluation on.
        top_n (int, optional): The number of top predictions to consider. Defaults to 5.
        print_freq (int, optional): The frequency of printing the evaluation progress. Defaults to 100.
        log_suffix (str, optional): The suffix to add to the evaluation log. Defaults to "".
        actual_target (int, optional): The actual target class to use for evaluation. Defaults to None.

    Returns:
        tuple: A tuple containing the evaluation metrics, the top incorrect classes, and the mean confidence.
            - metrics (dict): A dictionary containing the evaluation metrics (e.g., accuracy@1, accuracy@5).
            - top_n_incorrect_classes (list): A list of the top incorrect classes.
            - conf_mean (float): The mean confidence of correct predictions.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"
    if num_classes < top_n:
        print(f"Top K is not meaningful for few classes. Setting topk to {num_classes}")
        topk = (1, num_classes)
    else:
        topk = (1, top_n)
    num_processed_samples = 0
    top_prob_incorrect_classes_counter = (
        Counter()
    )  # memoize top_n incorrect classes and their counts
    correct_conf = 0
    correct_conf_count = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            if actual_target is not None:
                target = torch.tensor([actual_target] * image.shape[0]).to(
                    device, non_blocking=True
                )
            else:
                target = target.to(device, non_blocking=True)
            output = model(image)

            probs = torch.nn.functional.softmax(output, dim=1)
            # confidence
            conf, pred = torch.max(probs, dim=1)
            correct_conf_tensor = conf[pred == target]
            correct_conf += correct_conf_tensor.sum().item()
            correct_conf_count += correct_conf_tensor.shape[0]
            top_n_incorrect = [
                (
                    sorted(
                        ((idx, val) for idx, val in enumerate(xi) if idx != yi.item()),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:top_n]
                )
                for xi, yi in zip(probs.detach().cpu().numpy(), target)
            ]
            top_prob_incorrect_classes_counter.update(
                [k for i in top_n_incorrect for k, v in i]
            )
            acc1, acc5 = accuracy(output, target, topk=topk)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    top_n_incorrect_classes = sorted(
        top_prob_incorrect_classes_counter,
        key=lambda x: top_prob_incorrect_classes_counter[x],
        reverse=True,
    )[:top_n]
    # gather the stats from all processes

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    metrics = dict()
    metrics["acc1"] = metric_logger.acc1.global_avg
    metrics["acc5"] = metric_logger.acc5.global_avg
    if correct_conf_count == 0:
        conf_mean = 0
    else:
        conf_mean = correct_conf / correct_conf_count
    return metrics, top_n_incorrect_classes, conf_mean
