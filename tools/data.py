"""
Helper functions for data processing and caching.

Functions:
    - get_module
    - ClassificationPresetTrain
    - ClassificationPresetEval
    - cacheTrainData
    - cacheValData
    - loadData
    - cacheGenData
    - getSubsetIndicies
    - getSubsetsFromIndicies
    - getSubsets
    - cacheAllInterpolation
    - loadAllInterpolation
    - getSubsetsByInterpolation
    - getValSampler
    - getSubsetLoader
    - getSubsetLoaderbyInterpolation
    - mix_Data
    - synthetic_only_Data
"""

import os
import shutil

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Set default cache path

cache_path = "./data"

# Set default parameters for training and validation transformations

train_param = {
    "train_crop_size": 224,
    "interpolation": "bilinear",
    "auto_augment": None,
    "ra_magnitude": 9,
    "augmix_severity": 3,
    "random_erase_prob": 0.0,
    "backend": "pil",
    "use_v2": False,
}

val_param = {
    "val_resize_size": 256,
    "val_crop_size": 224,
    "interpolation": "bilinear",
    "backend": "pil",
    "use_v2": False,
}


def get_module(use_v2):
    """
    Returns the appropriate module from the torchvision.transforms package based on the value of use_v2.

    Parameters:
        use_v2 (bool): If True, returns the v2 module from torchvision.transforms.v2. If False, returns the module from torchvision.transforms.

    Returns:
        module: The selected module from torchvision.transforms or torchvision.transforms.v2.
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


class ClassificationPresetTrain:
    """
    A class representing a preset of transformations for training image classification models.

    Args:
        crop_size (int): The size of the cropped image.
        mean (tuple, optional): The mean values for normalization. Defaults to (0.485, 0.456, 0.406).
        std (tuple, optional): The standard deviation values for normalization. Defaults to (0.229, 0.224, 0.225).
        interpolation (InterpolationMode, optional): The interpolation mode for resizing. Defaults to InterpolationMode.BILINEAR.
        hflip_prob (float, optional): The probability of applying horizontal flip. Defaults to 0.5.
        auto_augment_policy (str, optional): The policy for applying auto augmentation. Defaults to None.
        ra_magnitude (int, optional): The magnitude for RandAugment policy. Defaults to 9.
        augmix_severity (int, optional): The severity for AugMix policy. Defaults to 3.
        random_erase_prob (float, optional): The probability of applying random erasing. Defaults to 0.0.
        backend (str, optional): The backend for image processing. Can be 'pil' or 'tensor'. Defaults to "pil".
        use_v2 (bool, optional): Whether to use version 2 of the transformations. Defaults to False.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################

    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(
            T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)
        )
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(
                    T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude)
                )
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(
                    T.AugMix(interpolation=interpolation, severity=augmix_severity)
                )
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(
                    T.AutoAugment(policy=aa_policy, interpolation=interpolation)
                )

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                (
                    T.ToDtype(torch.float, scale=True)
                    if use_v2
                    else T.ConvertImageDtype(torch.float)
                ),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    """
    A class representing a preset evaluation for classification tasks.

    Args:
        crop_size (int): The size of the cropped image.
        resize_size (int, optional): The size to resize the image to. Defaults to 256.
        mean (tuple, optional): The mean values for image normalization. Defaults to (0.485, 0.456, 0.406).
        std (tuple, optional): The standard deviation values for image normalization. Defaults to (0.229, 0.224, 0.225).
        interpolation (InterpolationMode, optional): The interpolation mode for resizing the image. Defaults to InterpolationMode.BILINEAR.
        backend (str, optional): The backend library to use for image processing. Can be 'tensor' or 'pil'. Defaults to "pil".
        use_v2 (bool, optional): Whether to use the v2 version of the transforms. Defaults to False.
    """

    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################

    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            (
                T.ToDtype(torch.float, scale=True)
                if use_v2
                else T.ConvertImageDtype(torch.float)
            ),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


def cacheTrainData(
    train_dir,
    ds_name,
    save_path=cache_path,
    default_param=train_param,
    stats={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
):
    """
    Caches the ImageNet dataset for training.

    Args:
        train_dir (str): The directory path containing the training images.
        ds_name (str): The name of the dataset.
        save_path (str, optional): The directory path to save the cached dataset. Defaults to cache_path.
        default_param (dict, optional): The default parameters for the dataset. Defaults to train_param.
        stats (dict, optional): The statistics for normalization. Defaults to {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}.

    Returns:
        None
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################

    # Load the ImageNet dataset statistics
    mean = stats["mean"]
    std = stats["std"]

    dataset = torchvision.datasets.ImageFolder(
        train_dir,
        ClassificationPresetTrain(
            crop_size=default_param["train_crop_size"],
            mean=mean,
            std=std,
            interpolation=transforms.functional.InterpolationMode(
                default_param["interpolation"]
            ),
            auto_augment_policy=default_param["auto_augment"],
            random_erase_prob=default_param["random_erase_prob"],
            ra_magnitude=default_param["ra_magnitude"],
            augmix_severity=default_param["augmix_severity"],
            backend=default_param["backend"],
            use_v2=default_param["use_v2"],
        ),
    )

    # Create directory if not exist
    os.makedirs(save_path, exist_ok=True)
    # Save the dataset to tensor file
    torch.save(dataset, os.path.join(save_path, f"{ds_name}.pt"))
    print(f"{ds_name}.pt dataset saved to {save_path}")
    return None


def cacheValData(
    valdir,
    ds_name,
    save_path=cache_path,
    default_param=val_param,
    stats={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
):
    """
    Caches the ImageNet validation dataset to a tensor file.

    Args:
        valdir (str): Path to the validation dataset directory.
        ds_name (str): Name of the dataset.
        save_path (str, optional): Path to save the cached dataset. Defaults to cache_path.
        default_param (dict, optional): Default parameters for dataset creation. Defaults to val_param.
        stats (dict, optional): Dictionary containing the mean and standard deviation values for normalization.
                               Defaults to {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}.

    Returns:
        None
    """
    ############################
    # Reference:
    # TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
    ############################
    mean = stats["mean"]
    std = stats["std"]
    # Load the ImageNet dataset
    dataset = torchvision.datasets.ImageFolder(
        valdir,
        ClassificationPresetEval(
            resize_size=default_param["val_resize_size"],
            crop_size=default_param["val_crop_size"],
            mean=mean,
            std=std,
            interpolation=transforms.functional.InterpolationMode(
                default_param["interpolation"]
            ),
            backend=default_param["backend"],
            use_v2=default_param["use_v2"],
        ),
    )

    # Create directory if not exist
    os.makedirs(save_path, exist_ok=True)
    # Save the dataset to tensor file
    torch.save(dataset, os.path.join(save_path, f"{ds_name}.pt"))
    print(f"{ds_name}.pt dataset saved to {save_path}")
    return None


def loadData(file_name, cache_path=cache_path):
    """
    Load a dataset from a cache path if it exists, otherwise print a message indicating that the dataset was not found.

    Args:
        file_name (str): The name of the dataset file.
        cache_path (str, optional): The path to the cache directory. Defaults to cache_path.

    Returns:
        dataset: The loaded dataset if it exists, otherwise None.
    """
    if os.path.exists(cache_path):
        dataset = torch.load(
            os.path.join(cache_path, f"{file_name}.pt"), weights_only=False
        )
        print(f"{file_name}.pt dataset loaded from {cache_path}")
    else:
        print(f"{file_name}.pt not found in {cache_path}")
        dataset = None
    return dataset


def cacheGenData(
    genInput_dir,
    ds_name,
    save_path=cache_path,
    resize=(256, 256),
    do_rescale=True,
):
    """
    Caches the ImageNet dataset generated from the given directory.

    Args:
        genInput_dir (str): The directory containing the generated dataset.
        ds_name (str): The name of the dataset.
        save_path (str, optional): The path to save the cached dataset. Defaults to cache_path.
        resize (tuple, optional): The size to resize the images to. Defaults to (256, 256).
        do_rescale (bool, optional): Whether to rescale the images or not. Defaults to True.

    Returns:
        None
    """
    # Load the ImageNet dataset
    if do_rescale:
        dataset = torchvision.datasets.ImageFolder(
            genInput_dir,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(resize)]
            ),
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            genInput_dir, transform=transforms.Compose([transforms.PILToTensor()])
        )

    # Create directory if not exist
    os.makedirs(save_path, exist_ok=True)
    # Save the dataset to tensor file
    torch.save(dataset, os.path.join(save_path, f"{ds_name}.pt"))
    print(f"{ds_name}.pt dataset saved to {save_path}")
    return None


def getSubsetIndicies(dataset, data_dir):
    """
    Returns a dictionary containing the start and end indices for each class in the dataset.

    Args:
        dataset (torchvision.datasets.Dataset): The dataset object.
        data_dir (str): The directory path where the dataset is stored.

    Returns:
        dict: A dictionary containing the start and end indices for each class in the dataset.
              The keys are the class names and the values are tuples of start and end indices.
    """
    synth_counts = {
        i: len(os.listdir(os.path.join(data_dir, i))) for i in os.listdir(data_dir)
    }
    synth_indices_fib = [0]
    for n, _ in dataset.class_to_idx.items():
        synth_indices_fib.append(synth_indices_fib[-1] + synth_counts[n])
    synth_indices = {
        k: (startcount, endcount)
        for k, startcount, endcount in zip(
            dataset.class_to_idx.keys(),
            synth_indices_fib,
            synth_indices_fib[1::],
        )
    }

    return synth_indices


def getSubsetsFromIndicies(dataset, indices):
    """
    Create subsets from a dataset based on given indices.

    Args:
        dataset (list): The dataset from which to create subsets.
        indices (dict): A dictionary containing the indices for each subset.

    Returns:
        dict: A dictionary containing the subsets created from the dataset.

    """
    subsets = {k: Subset(dataset, range(*v)) for k, v in indices.items()}
    return subsets


def getSubsets(dataset: torch.utils.data.Dataset, data_dir: str):
    """Get subsets of the dataset based on the number of synthetic images per class.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to subset.
        data_dir (str): The directory containing the synthetic images.

    Returns:
        dict: A dictionary of subsets.
    """
    synth_counts = {
        i: len(os.listdir(os.path.join(data_dir, i))) for i in os.listdir(data_dir)
    }
    synth_indices_fib = [0]
    for n, _ in dataset.class_to_idx.items():
        synth_indices_fib.append(synth_indices_fib[-1] + synth_counts[n])
    synth_indices = {
        k: (startcount, endcount)
        for k, startcount, endcount in zip(
            dataset.class_to_idx.keys(),
            synth_indices_fib,
            synth_indices_fib[1::],
        )
    }
    subsets = {k: Subset(dataset, range(*v)) for k, v in synth_indices.items()}
    return subsets


def cacheAllInterpolation(
    data_path,
    save_path,
    prefix="synth",
    resize=(256, 256),
    do_rescale=True,
    cache_val=False,
):
    """
    Caches all interpolation data for each class in the given data path.

    Args:
        data_path (str): The path to the directory containing the data.
        save_path (str): The path to save the cached data.
        prefix (str, optional): The prefix to use for the cached data files. Defaults to "synth".
        resize (tuple, optional): The size to resize the data to. Defaults to (256, 256).
        do_rescale (bool, optional): Whether to rescale the data. Defaults to True.
        cache_val (bool, optional): Whether to cache validation data. Defaults to False.
    """
    data_classes = sorted(os.listdir(data_path))
    for c in data_classes:
        sub_path = os.path.join(data_path, c)
        if os.path.isdir(sub_path):
            if cache_val:
                cacheValData(sub_path, f"{prefix}_{c}", save_path=save_path)
            else:
                cacheGenData(
                    sub_path,
                    f"{prefix}_{c}",
                    save_path=save_path,
                    resize=resize,
                    do_rescale=do_rescale,
                )


def loadAllInterpolation(data_path, cache_path, prefix="synth"):
    """
    Load all interpolation data from the specified data path.

    Args:
        data_path (str): The path to the data directory.
        cache_path (str): The path to the cache directory.
        prefix (str, optional): The prefix to use for the loaded data. Defaults to "synth".

    Returns:
        dict: A dictionary containing the loaded interpolation data, where the keys are the class names and the values are the loaded data.

    """
    data_classes = sorted(os.listdir(data_path))
    dataset_classes = dict()
    for c in data_classes:
        sub_path = os.path.join(data_path, c)
        if os.path.isdir(sub_path):
            dataset_classes[c] = loadData(
                f"{prefix}_{c}",
                cache_path=cache_path,
            )
    return dataset_classes


def getSubsetsByInterpolation(data_path, cache_path, prefix="synth"):
    """
    Retrieves subsets of interpolated datasets.

    Args:
        data_path (str): The path to the data directory.
        cache_path (str): The path to the cache directory.
        prefix (str, optional): The prefix for the dataset files. Defaults to "synth".

    Returns:
        dict: A dictionary containing subsets of interpolated datasets.
    """
    subset_interpolation = dict()
    interpolate_datasets = loadAllInterpolation(data_path, cache_path, prefix=prefix)
    for k, v in interpolate_datasets.items():
        class_path = os.path.join(data_path, k)
        subset_interpolation[k] = getSubsets(v, class_path)
    return subset_interpolation


def getValSampler(val_dataset, distributed=False):
    """
    Returns a sampler for the validation dataset.

    Args:
        val_dataset (torch.utils.data.Dataset): The validation dataset.
        distributed (bool, optional): Whether to use distributed sampling.
            Defaults to False.

    Returns:
        torch.utils.data.Sampler: The sampler for the validation dataset.
    """
    if distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    return val_sampler


def getSubsetLoader(
    subsets,
    batch_size=4,
    num_workers=1,
    pin_memory=True,
    shuffle=False,
    val_loader=False,
    distributed=False,
    collate_fn=torch.utils.data.dataloader.default_collate,
):
    """
    Returns a dictionary of data loaders for the given subsets.

    Args:
        subsets (dict): A dictionary containing the subsets of data.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 4.
        num_workers (int, optional): The number of worker threads to use for loading the data. Defaults to 1.
        pin_memory (bool, optional): If True, the data loader will copy tensors into pinned memory. Defaults to True.
        shuffle (bool, optional): If True, the data will be shuffled before each epoch. Defaults to False.
        val_loader (bool, optional): If True, the data loaders will be created for validation. Defaults to False.
        distributed (bool, optional): If True, the data will be loaded in a distributed manner. Defaults to False.
        collate_fn (callable, optional): The function used to collate the data samples. Defaults to torch.utils.data.dataloader.default_collate.

    Returns:
        dict: A dictionary containing the data loaders for each subset.
    """
    if val_loader:
        return {
            target: DataLoader(
                subset,
                batch_size=batch_size,
                sampler=getValSampler(subset, distributed=distributed),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
            for target, subset in subsets.items()
        }
    return {
        target: DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )
        for target, subset in subsets.items()
    }


def getSubsetLoaderbyInterpolation(
    subsetsByInterpolation,
    batch_size=4,
    num_workers=1,
    pin_memory=True,
    shuffle=False,
    val_loader=False,
    distributed=False,
    collate_fn=torch.utils.data.dataloader.default_collate,
):
    """
    Returns a dictionary of subset loaders by interpolation.

    Args:
        subsetsByInterpolation (dict): A dictionary containing subsets of data by interpolation.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 4.
        num_workers (int, optional): The number of worker threads for data loading. Defaults to 1.
        pin_memory (bool, optional): If True, the data loader will copy tensors into pinned memory. Defaults to True.
        shuffle (bool, optional): If True, the data will be shuffled at every epoch. Defaults to False.
        val_loader (bool, optional): If True, the data loader will be used for validation. Defaults to False.
        distributed (bool, optional): If True, the data will be distributed across multiple devices. Defaults to False.
        collate_fn (callable, optional): The function used to collate the data samples. Defaults to torch.utils.data.dataloader.default_collate.

    Returns:
        dict: A dictionary of subset loaders by interpolation.
    """
    return {
        k: getSubsetLoader(
            v,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            val_loader=val_loader,
            distributed=distributed,
            collate_fn=collate_fn,
        )
        for k, v in subsetsByInterpolation.items()
    }


def mix_Data(original_dir, overwrite=False, create_suffix="mix"):
    """
    Mixes the data in the original directory by moving files from the 'synthetic' subdirectory to the 'train' subdirectory.

    Args:
        original_dir (str): The path to the original directory containing the data.
        overwrite (bool, optional): Specifies whether to overwrite the original directory or create a new directory with a suffix. Defaults to False.
        create_suffix (str, optional): The suffix to be added to the original directory name when creating a new directory. Defaults to "mix".

    Returns:
        str: The path to the mixed data directory.

    Raises:
        FileExistsError: If the new directory already exists and overwrite is set to False.

    """
    if not overwrite:
        mix_dir = f"{original_dir}-{create_suffix}"
        if not os.path.exists(mix_dir):
            shutil.copytree(original_dir, mix_dir)
        else:
            raise FileExistsError(f"{mix_dir} already exists")
    else:
        mix_dir = original_dir

    synth_dir = f"{mix_dir}/synthetic"
    train_dir = f"{mix_dir}/train"

    for root, dirs, files in os.walk(synth_dir):
        for file in files:
            dest_folder = os.path.join(train_dir, os.path.basename(root))
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            os.rename(os.path.join(root, file), os.path.join(dest_folder, file))
    shutil.rmtree(synth_dir)

    return mix_dir


def synthetic_only_Data(original_dir, overwrite=False, create_suffix="synthetic"):
    """
    Create a synthetic-only dataset by copying the original directory and renaming it with a suffix.
    If `overwrite` is False, a new directory with the suffix will be created and the original directory will be copied into it.
    If `overwrite` is True, the original directory will be used as the synthetic directory.
    The 'train' subdirectory inside the synthetic directory will be renamed to 'synthetic'.

    Args:
        original_dir (str): The path to the original directory.
        overwrite (bool, optional): Whether to overwrite the original directory. Defaults to False.
        create_suffix (str, optional): The suffix to add to the original directory name. Defaults to "synthetic".

    Returns:
        str: The path to the synthetic directory.

    Raises:
        FileExistsError: If the synthetic directory already exists and `overwrite` is False.
    """
    if not overwrite:
        synth_dir = f"{original_dir}-{create_suffix}"
        if not os.path.exists(synth_dir):
            shutil.copytree(original_dir, synth_dir)
        else:
            raise FileExistsError(f"{synth_dir} already exists")
    else:
        synth_dir = original_dir

    shutil.rmtree(f"{synth_dir}/train")
    os.rename(f"{synth_dir}/synthetic", f"{synth_dir}/train")

    return synth_dir
