# ---------------------------------------------------------------------
# To run this script, call `python -m scripts.train` in the terminal
#
# Notes:
# For DATASET_DIR, the path starts from project directory
# Ensure that file names are identical for a pair of image and image mask
# ---------------------------------------------------------------------
import json
import os
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from diunet import DIUNet
from utils import ImageSegmentationDataset

# initialize run name
RUN_NAME = str(datetime.now().strftime("run_%d-%m-%Y_%H-%M"))
if not os.path.exists(f"./logs/{RUN_NAME}"):
    os.makedirs(f"./logs/{RUN_NAME}")

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
MODEL_CHANNEL_SCALE = 0.1
DENSE_BLOCK_DEPTH_SCALE = 0.1
model = DIUNet(
    channel_scale=MODEL_CHANNEL_SCALE, dense_block_depth_scale=DENSE_BLOCK_DEPTH_SCALE
)
model.to(device)
print(f"Info: Model loaded has {sum(p.numel() for p in model.parameters())} parameters")

# training configuration and hyperparameters
DATASET_DIR = "./data/model_training"
EPOCHS = 3
BATCH_SIZE = 8

optimizer = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
loss_fn = nn.BCEWithLogitsLoss()

# transformation for images
transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.to(device)),
    ]
)

# create dataset
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
test_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/test/images",
    f"{DATASET_DIR}/test/image_masks",
    transforms,
    transforms,
)

# create dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# data structure for logging
log = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "model_channel_scale": MODEL_CHANNEL_SCALE,
    "dense_block_depth_scale": DENSE_BLOCK_DEPTH_SCALE,
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "train_loss": [],
    "val_loss": [],
}

# currently saves best model based on validation BCE loss
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # start training loop
    model.train()
    train_loss_sum = 0
    train_running_loss = 0

    for train_batch_idx, (train_imgs, train_img_masks) in enumerate(
        pbar := tqdm(train_dataloader)
    ):
        pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {train_running_loss}")

        train_preds = model(train_imgs)
        loss: torch.Tensor = loss_fn(train_preds, train_img_masks)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_sum += loss.item()
        train_running_loss = train_loss_sum / (train_batch_idx + 1)

    log["train_loss"].append(train_running_loss)

    # start evaluation loop
    model.eval()
    val_loss_sum = 0
    val_running_loss = 0

    for val_batch_idx, (val_imgs, val_img_masks) in enumerate(
        pbar := tqdm(val_dataloader)
    ):
        pbar.set_description(f"Epoch: {epoch+1}, Val Loss: {val_running_loss}")

        with torch.no_grad():
            val_preds = model(val_imgs)
            loss = loss_fn(val_preds, val_img_masks)

        val_loss_sum += loss.item()
        val_running_loss = val_loss_sum / (val_batch_idx + 1)

    log["val_loss"].append(val_running_loss)

    if val_running_loss < best_val_loss:
        best_val_loss = val_running_loss
        torch.save(model.state_dict(), f"./logs/{RUN_NAME}/best_model_state_dict.pt")

# save log
with open(f"./logs/{RUN_NAME}/results.json", "w") as outfile:
    json.dump(log, outfile)
