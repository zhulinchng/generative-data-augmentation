"""
Train a classifier on specified dataset with PyTorch.

Adapted from TorchVision reference scripts with additional features:
- ScheduleFree Optimizer for improved convergence
- Weights & Biases integration for experiment tracking
- MultiStepLR scheduler with configurable milestones

Reference:
TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'.
GitHub. Available at: https://github.com/pytorch/vision.
"""

import torch
import torchvision

from tools import data, utils
from tools import transforms as trnf


def main():
    """Main training function."""
    # Enable performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True

    # Training configuration
    setup = {
        # Experiment tracking
        "entity": "zhulinchng",  # Wandb username
        "project": "demo",  # Wandb project name
        "wandb_id": "",
        # Dataset configuration
        "data_path": "./data/imagenette",  # imagenette, woof, or stanford-dogs
        "cache_dataset": False,
        "workers": 12,  # Number of data loading workers
        # Model configuration
        "model": "resnet18",  # resnet18 or mobilenet_v2
        "weights": None,  # Path to pretrained weights or None
        "device": "cuda",
        # Training parameters
        "epochs": 200,
        "batch_size": 256,  # 256 for resnet18, 128 for mobilenet_v2 (12GB GPU)
        "start_epoch": 0,
        "resume": "",  # Path to checkpoint to resume from
        "test_only": False,
        # Optimizer configuration
        "opt": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "norm_weight_decay": None,
        "bias_weight_decay": None,
        "transformer_embedding_decay": None,
        # Learning rate schedule
        "lr_scheduler": "multisteplr",
        "lr_milestones": [100, 150, 201, 251],
        "lr_gamma": 0.1,
        "lr_min": 0.0,
        "lr_step_size": 50,
        "lr_warmup_epochs": 0,
        "lr_warmup_method": "constant",
        "lr_warmup_decay": 0.01,
        # Data augmentation
        "auto_augment": None,
        "ra_magnitude": 9,
        "augmix_severity": 3,
        "random_erase": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "label_smoothing": 0.0,
        # Training settings
        "interpolation": "bilinear",
        "val_resize_size": 256,
        "val_crop_size": 224,
        "train_crop_size": 224,
        "clip_grad_norm": None,
        "print_freq": 20,
        "output_dir": "./output/imagenette",
        # Advanced features
        "amp": False,  # Automatic mixed precision
        "model_ema": False,  # Exponential moving average
        "model_ema_steps": 32,
        "model_ema_decay": 0.99998,
        "sync_bn": False,
        "use_deterministic_algorithms": False,
        # Distributed training
        "world_size": 1,
        "dist_url": "env://",
        # Miscellaneous
        "ra_sampler": False,
        "ra_reps": 3,
        "backend": "PIL",
        "use_v2": False,
    }

    # Prepare dataset paths
    train_dir = f'{setup["data_path"].replace("./","")}/train'
    val_dir = f'{setup["data_path"].replace("./","")}/val'

    # Cache datasets for faster loading
    data.cacheTrainData(train_dir, "train_cache", save_path=setup["data_path"])
    data.cacheValData(val_dir, "val_cache", save_path=setup["data_path"])

    # Initialize distributed training
    setup = utils.init_distributed_mode(setup)
    device = torch.device(setup["device"])

    # Configure deterministic behavior
    if setup["use_deterministic_algorithms"]:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # Load and validate training dataset
    train_dataset = data.loadData("train_cache", cache_path=setup["data_path"])
    num_classes = len(train_dataset.classes)
    assert (
        num_classes == 10
    ), f"Expected 10 classes, got {num_classes}"  # 10 for imagenette/woof, 120 for stanford-dogs

    # Create training data loader
    train_sampler = utils.getTrainSampler(train_dataset, setup)
    collate_fn = trnf.getCollateFn(num_classes=num_classes, setup=setup)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=setup["batch_size"],
        sampler=train_sampler,
        num_workers=setup["workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load and validate validation dataset
    val_dataset = data.loadData("val_cache", cache_path=setup["data_path"])
    assert (
        len(val_dataset.classes) == 10
    ), f"Expected 10 classes, got {len(val_dataset.classes)}"

    # Create validation data loader
    val_sampler = utils.getValSampler(val_dataset, setup)
    data_loader_test = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=setup["batch_size"],
        sampler=val_sampler,
        num_workers=setup["workers"],
        pin_memory=True,
        shuffle=False,
    )

    # Initialize model
    model = torchvision.models.get_model(
        setup["model"], weights=setup["weights"], num_classes=num_classes
    )
    model.to(device)

    # Apply synchronized batch normalization if distributed
    if setup["distributed"] and setup["sync_bn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Setup training components
    training_components = utils.trainSetup(
        model=model, setup=setup, val_dataloader=data_loader_test, device=device
    )
    (
        model,
        criterion,
        optimizer,
        setup,
        model_ema,
        scaler,
        model_without_ddp,
        lr_scheduler,
        run,
    ) = training_components

    # Train the model
    model, run = utils.train(
        model,
        model_without_ddp,
        criterion,
        optimizer,
        lr_scheduler,
        data_loader,
        train_sampler,
        data_loader_test,
        device,
        setup,
        model_ema,
        scaler,
        run,
    )

    # Final evaluation
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model.eval()

    metrics = utils.evaluate(
        model,
        criterion,
        data_loader_test,
        device=device,
        print_freq=setup["print_freq"],
    )

    # Log final metrics to wandb
    run.summary["acc1"] = metrics["acc1"]
    run.summary["acc5"] = metrics["acc5"]
    run.summary["loss"] = metrics["loss"]
    run.finish()


if __name__ == "__main__":
    main()
