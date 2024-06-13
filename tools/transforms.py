"""
Helper script to implement MixUp and CutMix augmentations.
The script is based on the implementation in the TorchVision library.

Functions:
    - get_module
    - get_mixup_cutmix
    - RandomMixUp
    - RandomCutMix
    - getCollateFn
"""

############################
# Reference:
# TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
############################

import math
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F


def get_module(use_v2):
    """
    Returns the appropriate module for torchvision transforms based on the value of `use_v2`.

    Parameters:
    - use_v2 (bool): If True, returns the v2 module of torchvision.transforms. If False, returns the default module.

    Returns:
    - module: The module for torchvision transforms.

    References:
    - TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


def get_mixup_cutmix(*, mixup_alpha, cutmix_alpha, num_classes, use_v2):
    """
    Returns a random transformation from MixUp and CutMix based on the provided parameters.

    Args:
        mixup_alpha (float): The alpha value for MixUp transformation.
        cutmix_alpha (float): The alpha value for CutMix transformation.
        num_classes (int): The number of classes in the dataset.
        use_v2 (bool): A flag indicating whether to use the v2 version of the transforms module.

    Returns:
        A random transformation from MixUp and CutMix, or None if both mixup_alpha and cutmix_alpha are 0.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    transforms_module = get_module(use_v2)

    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(
            transforms_module.MixUp(alpha=mixup_alpha, num_classes=num_classes)
            if use_v2
            else RandomMixUp(num_classes=num_classes, p=1.0, alpha=mixup_alpha)
        )
    if cutmix_alpha > 0:
        mixup_cutmix.append(
            transforms_module.CutMix(alpha=mixup_alpha, num_classes=num_classes)
            if use_v2
            else RandomCutMix(num_classes=num_classes, p=1.0, alpha=mixup_alpha)
        )
    if not mixup_cutmix:
        return None

    return transforms_module.RandomChoice(mixup_cutmix)


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


def getCollateFn(num_classes, setup: dict, get_mixup_cutmix=get_mixup_cutmix):
    """
    Returns a collate function for data batching that applies mixup or cutmix transformations.

    Args:
        num_classes (int): The number of classes in the dataset.
        setup (dict): A dictionary containing setup parameters for mixup and cutmix.
        get_mixup_cutmix (function, optional): A function that returns mixup or cutmix transformation.
            Defaults to get_mixup_cutmix.

    Returns:
        collate_fn (function): The collate function for data batching.
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=setup["mixup_alpha"],
        cutmix_alpha=setup["cutmix_alpha"],
        num_classes=num_classes,
        use_v2=setup["use_v2"],
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    return collate_fn
