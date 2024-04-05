# ---------------------------------------------------------------------
# To run this script, call `python -m scripts.train` in the terminal
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
from torchvision.transforms import v2

from diunet import DIUNet
from utils import ImageSegmentationDataset, EarlyStopper, Logger


PARAMS = {
    "max_epochs": 3,
    "batch_size": 8,
    "model_channel_scale": 0.25,
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
print(f"Info: Model loaded has {sum(p.numel() for p in model.parameters())} parameters")

# training configuration and hyperparameters
DATASET_DIR = "./data/model_training"
optimizer = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
loss_fn = nn.BCEWithLogitsLoss()

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
train_dataloader = DataLoader(
    train_dataset, batch_size=PARAMS["batch_size"], shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=PARAMS["batch_size"])
test_dataloader = DataLoader(test_dataset, batch_size=PARAMS["batch_size"])

# ---------------------------------------------
# Model training
# ---------------------------------------------

logger = Logger(PARAMS)
early_stopper = EarlyStopper(patience=10, min_delta=0.03)


# currently saves best model based on validation BCE loss
best_val_loss = float("inf")

for epoch in range(PARAMS["max_epochs"]):
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

    # log results
    logger.train_loss.append(train_running_loss)
    logger.val_loss.append(val_running_loss)

    # save best model
    if val_running_loss < best_val_loss:
        best_val_loss = val_running_loss
        torch.save(
            model.state_dict(), f"./logs/{logger.run_name}/best_model_state_dict.pt"
        )

    # early stopping
    if early_stopper.early_stop(val_running_loss):
        break

# save log
logger.epochs_trained = epoch + 1
logger.save_run()
