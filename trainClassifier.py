"""
This script trains a classifier on a specified dataset with PyTorch.

Most of the code is adapted from the TorchVision reference scripts:
https://github.com/pytorch/vision/tree/main/references/classification

The script was modified to include additional features such as:
- ScheduleFree Optimizer (https://arxiv.org/abs/2405.15682)
- Experiment Tracking with Weights and Biases (https://wandb.ai/)
- Learning Rate Milestones (MultiStepLR) for LR Scheduler (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html)

During the development of the script, a contribution was made to the TorchVision repository:
- Title: Fix CutMix and MixUp arguments in transforms.py
- Description: Fixed the CutMix and MixUp arguments in the transforms.py reference script to train a classifier.
- Commit: https://github.com/pytorch/vision/commit/3b5e6fc42403cc20f8cfbafb52a75b30bd00c226
- Pull Request: https://github.com/pytorch/vision/pull/8287 

Reference:
TorchVision maintainers and contributors (2016) 'TorchVision: PyTorch's computer vision library'. GitHub. Available at: https://github.com/pytorch/vision.
"""

import torch
import torchvision

from tools import data, utils
from tools import transforms as trnf

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True  # enable TensorFloat32 for matmul
    setup = {
        "entity": "zhulinchng",  # the wandb username
        "project": "demo",  # the wandb project name
        "data_path": "./data/imagenette",  # ./data/imagenette for imagenette; ./data/woof for imagewoof; ./stanford-dogs for stanford-dogs
        "device": "cuda",
        "model": "resnet18",  # resnet18, mobilenet_v2
        "weights": None,  # path to model weights, None to train from scratch
        "batch_size": 256,  # batch size 256 for resnet18, 128 for mobilenet_v2 to fit 12GB of GPU memory optimally
        "epochs": 200,
        "workers": 12,  # number of data loading workers, CPU cores
        "opt": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "norm_weight_decay": None,
        "bias_weight_decay": None,
        "transformer_embedding_decay": None,
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "cutmix_alpha": 0.0,
        "lr_scheduler": "multisteplr",
        "lr_warmup_epochs": 0,
        "lr_warmup_method": "constant",
        "lr_warmup_decay": 0.01,
        "lr_step_size": 50,
        "lr_gamma": 0.1,
        "lr_min": 0.0,
        "lr_milestones": [
            100,
            150,
            201,
            251,
        ],  # if using multisteplr, else milestones are not used
        "print_freq": 20,
        "output_dir": "./output/imagenette",  # output directory for model checkpoints
        "resume": "",  # path to model to resume training, empty to train from scratch
        "start_epoch": 0,
        "cache_dataset": False,
        "sync_bn": False,
        "test_only": False,  # set to True to only test the model
        "auto_augment": None,
        "ra_magnitude": 9,
        "augmix_severity": 3,
        "random_erase": 0.0,
        "amp": False,
        "world_size": 1,
        "dist_url": "env://",
        "model_ema": False,
        "model_ema_steps": 32,
        "model_ema_decay": 0.99998,
        "use_deterministic_algorithms": False,
        "interpolation": "bilinear",
        "val_resize_size": 256,
        "val_crop_size": 224,
        "train_crop_size": 224,
        "clip_grad_norm": None,
        "ra_sampler": False,
        "ra_reps": 3,
        "backend": "PIL",
        "use_v2": False,
        "wandb_id": "",
    }

    ### Cache Dataset for Faster Loading
    train_dir = f'{setup["data_path"].replace("./","")}/train'
    valdir = f'{setup["data_path"].replace("./","")}/val'

    data.cacheTrainData(train_dir, "train_cache", save_path=setup["data_path"])
    data.cacheValData(valdir, "val_cache", save_path=setup["data_path"])

    ### Training

    setup = utils.init_distributed_mode(setup)
    device = torch.device(setup["device"])
    if setup["use_deterministic_algorithms"]:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    # Load training set
    train_dataset = data.loadData("train_cache", cache_path=setup["data_path"])
    num_classes = len(train_dataset.classes)

    # Check if the number of classes is correct
    assert num_classes == 10  # 10 for imagenette and imagewoof , 120 for stanford-dogs

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
    val_dataset = data.loadData("val_cache", cache_path=setup["data_path"])

    # Check if the number of classes is correct
    # 10 for imagenette and imagewoof , 120 for stanford-dogs
    assert len(val_dataset.classes) == 10

    val_sampler = utils.getValSampler(val_dataset, setup)
    data_loader_test = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=setup["batch_size"],
        sampler=val_sampler,
        num_workers=setup["workers"],
        pin_memory=True,
        shuffle=False,
    )

    model = torchvision.models.get_model(
        setup["model"], weights=setup["weights"], num_classes=num_classes
    )
    model.to(device)
    if setup["distributed"] and setup["sync_bn"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Train Setup
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
    ) = utils.trainSetup(
        model=model, setup=setup, val_dataloader=data_loader_test, device=device
    )

    # Train
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

    # Evaluate
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

    # Log metrics to wandb
    run.summary["acc1"] = metrics["acc1"]
    run.summary["acc5"] = metrics["acc5"]
    run.summary["loss"] = metrics["loss"]
    run.finish()
