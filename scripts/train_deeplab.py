# ---------------------------------------------------------------------
# To run this script, run `python -m scripts.train_deeplab`
# To monitor Tensorboard, run `tensorboard serve --logdir runs/`
#
# Notes:
# For DATASET_DIR, the path starts from project directory
# Ensure that file names are identical for a pair of image and image mask
# ---------------------------------------------------------------------
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import v2

from utils import ImageSegmentationDataset, Logger, BinaryMIOU
from deeplab import inception_deeplabv3

# ---------------------------------------------
# Training preparation
# ---------------------------------------------
DATASET_DIR = "./data/model_training"
TRAIN_DATA_DIR = "train_augmented_v2"
VAL_DATA_DIR = "val_augmented_v2"

PARAMS = {
    "model_name": "3module-inception-deeplabv3-aug-v2",
    "max_epochs": 45,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "backbone": "resnet50",
    "inception_modules": 3,
}

# Check GPU availability
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(
        f"Info: CUDA GPU detected, using {torch.cuda.get_device_name(torch.cuda.current_device())} for training"
    )
else:
    device = torch.device("cpu")
    print("Info: CUDA GPU not detected, using CPU for training")

# model configuration
model = inception_deeplabv3(
    backbone=PARAMS["backbone"], inception_module_count=PARAMS["inception_modules"]
)

# put model on device
model.to(device)
PARAMS["parameter_count"] = sum(p.numel() for p in model.parameters())
print(f"Info: Model loaded has {PARAMS['parameter_count']} parameters")

# ---------------------------------------------
# Dataset preparation
# ---------------------------------------------

# transformation for images
transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.to(device)),
    ]
)

# create dataset
train_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/{TRAIN_DATA_DIR}/images",
    f"{DATASET_DIR}/{TRAIN_DATA_DIR}/image_masks",
    transforms,
    transforms,
)
val_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/{VAL_DATA_DIR}/images",
    f"{DATASET_DIR}/{VAL_DATA_DIR}/image_masks",
    transforms,
    transforms,
)

# create dataloader
train_dataloader = DataLoader(
    train_dataset, batch_size=PARAMS["batch_size"], shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=PARAMS["batch_size"])

# ---------------------------------------------
# Model training
# ---------------------------------------------
writer = SummaryWriter(comment=f"_{PARAMS['model_name']}")
logger = Logger(logdir=f"./{writer.get_logdir()}")

loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=PARAMS["learning_rate"], betas=(0.9, 0.999))
scheduler = ReduceLROnPlateau(optimizer, patience=10)

miou_metric = BinaryMIOU(device=device)
best_miou = float("-inf")

for epoch in range(PARAMS["max_epochs"]):
    metrics = {
        # train metrics
        "train_loss_sum": 0,
        "train_running_loss": 0,
        "train_miou_sum": 0,
        "train_running_miou": 0,
        # val metrics
        "val_loss_sum": 0,
        "val_running_loss": 0,
        "val_miou_sum": 0,
        "val_running_miou": 0,
    }

    # start training loop
    model.train()

    for train_batch_idx, (train_imgs, train_img_masks) in enumerate(
        pbar := tqdm(train_dataloader)
    ):
        train_preds = model(train_imgs)["out"]
        train_loss: torch.Tensor = loss_fn(train_preds, train_img_masks)

        # back propagation
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # calculate metrics
        metrics["train_loss_sum"] += train_loss.item()
        metrics["train_running_loss"] = metrics["train_loss_sum"] / (
            train_batch_idx + 1
        )

        metrics["train_miou_sum"] += miou_metric(train_preds, train_img_masks)
        metrics["train_running_miou"] = metrics["train_miou_sum"] / (
            train_batch_idx + 1
        )

        pbar.set_description(
            f"Epoch: {epoch+1}, Train Loss: {metrics['train_running_loss']}"
            + f", Train mIoU: {metrics['train_running_miou']}"
        )

    # start evaluation loop
    model.eval()

    for val_batch_idx, (val_imgs, val_img_masks) in enumerate(
        pbar := tqdm(val_dataloader)
    ):
        with torch.no_grad():
            val_preds = model(val_imgs)["out"]
            val_loss: torch.Tensor = loss_fn(val_preds, val_img_masks)

        # calculate metrics
        metrics["val_loss_sum"] += val_loss.item()
        metrics["val_running_loss"] = metrics["val_loss_sum"] / (val_batch_idx + 1)

        metrics["val_miou_sum"] += miou_metric(val_preds, val_img_masks)
        metrics["val_running_miou"] = metrics["val_miou_sum"] / (val_batch_idx + 1)

        pbar.set_description(
            f"Epoch: {epoch+1}, Val Loss: {metrics['val_running_loss']}"
            + f", Val mIoU: {metrics['val_running_miou']}"
        )

    # log results
    writer.add_scalar("loss/train", metrics["train_running_loss"], epoch + 1)
    writer.add_scalar("loss/val", metrics["val_running_loss"], epoch + 1)

    writer.add_scalar("mIoU/train", metrics["train_running_miou"], epoch + 1)
    writer.add_scalar("mIoU/val", metrics["val_running_miou"], epoch + 1)

    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)

    # log loss to learning rate scheduler
    scheduler.step(val_loss)

    # save best model
    if metrics["val_running_miou"] > best_miou:
        best_miou = metrics["val_running_miou"]
        PARAMS["best_epoch"] = epoch + 1
        torch.save(model.state_dict(), f"{logger.logdir}/best_model_state_dict.pt")

# save final log
writer.flush()
writer.close()

PARAMS["epochs_trained"] = epoch + 1
logger.save_run(PARAMS)
