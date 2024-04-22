# ---------------------------------------------------------------------
# To run this script, run `python -m scripts.train`
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import v2

from diunet import DIUNet
from utils import ImageSegmentationDataset, EarlyStopper, Logger, BinaryMIOU

# ---------------------------------------------
# Training preparation
# ---------------------------------------------
DATASET_DIR = "./data/model_training"
PARAMS = {
    "description": "DIU-Net trained on original data",
    "max_epochs": 120,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "model_channel_scale": 0.5,
    "dense_block_depth_scale": 0.25,
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
model = DIUNet(
    channel_scale=PARAMS["model_channel_scale"],
    dense_block_depth_scale=PARAMS["dense_block_depth_scale"],
)
model.to(device)
PARAMS["parameter_count"] = sum(p.numel() for p in model.parameters())
print(f"Info: Model loaded has {PARAMS['parameter_count']} parameters")

# training configuration and hyperparameters
# early_stopper = EarlyStopper(patience=PARAMS["patience"])
optimizer = Adam(model.parameters(), lr=PARAMS["learning_rate"], betas=(0.9, 0.999))
loss_fn = nn.BCELoss()
miou_metric = BinaryMIOU(device=device)

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

# create datasets
train_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/train/images",
    f"{DATASET_DIR}/train/image_masks",
    transforms,
    transforms,
)
val_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/val/images",
    f"{DATASET_DIR}/val/image_masks",
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
logger = Logger()
writer = SummaryWriter()
best_val_miou = float("-inf")

for epoch in range(PARAMS["max_epochs"]):
    # reset metrics
    metrics = {
        "train_loss_sum": 0,
        "train_running_loss": 0,
        "train_iou_sum": 0,
        "train_running_iou": 0,
        "val_loss_sum": 0,
        "val_running_loss": 0,
        "val_iou_sum": 0,
        "val_running_iou": 0,
    }

    # start training loop
    model.train()

    for train_batch_idx, (train_imgs, train_img_masks) in enumerate(
        pbar := tqdm(train_dataloader)
    ):
        train_preds = model(train_imgs)
        loss: torch.Tensor = loss_fn(train_preds, train_img_masks)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # calculate metrics
        metrics["train_loss_sum"] += loss.item()
        metrics["train_running_loss"] = metrics["train_loss_sum"] / (
            train_batch_idx + 1
        )

        metrics["train_iou_sum"] += miou_metric(train_preds, train_img_masks)
        metrics["train_running_iou"] = metrics["train_iou_sum"] / (train_batch_idx + 1)

        pbar.set_description(
            f"Epoch: {epoch+1}, Train Loss: {metrics['train_running_loss']}, Train mIoU: {metrics['train_running_iou']}"
        )

    # start evaluation loop
    model.eval()

    for val_batch_idx, (val_imgs, val_img_masks) in enumerate(
        pbar := tqdm(val_dataloader)
    ):
        with torch.no_grad():
            val_preds = model(val_imgs)
            loss = loss_fn(val_preds, val_img_masks)

        # calculate metrics
        metrics["val_loss_sum"] += loss.item()
        metrics["val_running_loss"] = metrics["val_loss_sum"] / (val_batch_idx + 1)

        metrics["val_iou_sum"] += miou_metric(val_preds, val_img_masks)
        metrics["val_running_iou"] = metrics["val_iou_sum"] / (val_batch_idx + 1)

        pbar.set_description(
            f"Epoch: {epoch+1}, Val Loss: {metrics['val_running_loss']}, Val mIoU: {metrics['val_running_iou']}"
        )

    # log results
    writer.add_scalar("loss/train", metrics["train_running_loss"], epoch)
    writer.add_scalar("loss/val", metrics["val_running_loss"], epoch)
    writer.add_scalar("mIoU/train", metrics["train_running_iou"], epoch)
    writer.add_scalar("mIoU/val", metrics["val_running_iou"], epoch)

    # save best model
    if metrics["val_running_miou"] > best_val_miou:
        best_val_miou = metrics["val_running_miou"]
        PARAMS["best_epoch"] = epoch + 1
        torch.save(
            model.state_dict(), f"./logs/{logger.run_name}/best_model_state_dict.pt"
        )

    # early stopping
    # if early_stopper.early_stop(metrics["val_running_loss"]):
    #     break

# save final log
writer.flush()
writer.close()

PARAMS["epochs_trained"] = epoch + 1
logger.save_run(PARAMS)
