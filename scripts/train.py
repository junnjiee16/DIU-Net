# ---------------------------------------------------------------------
# To run this script, call `python -m scripts.train` in the terminal
#
# Notes:
# For DATASET_DIR, the path starts from project directory
# Ensure that file names are identical for a pair of image and image mask
# ---------------------------------------------------------------------
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from diunet import DIUNet
from utils import ImageSegmentationDataset

# TODO: load on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Info: Using CUDA GPU for training")
else:
    device = torch.device("cpu")
    print("Info: CUDA GPU not available, defaulting to CPU for training")

# model configuration
model = DIUNet(channel_scale=0.1, dense_block_depth_scale=0.1)
model.to(device)
print(f"Info: Model parameter count: {sum(p.numel() for p in model.parameters())}")

# training configuration and hyperparameters
DATASET_DIR = "./data/model_training"
BATCH_SIZE = 8
EPOCHS = 3

optimizer = Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
loss_fn = CrossEntropyLoss()

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


for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0

    for train_batch_idx, (train_imgs, train_img_masks) in enumerate(
        tqdm(train_dataloader)
    ):
        train_preds = model(train_imgs)
        print(train_img_masks.shape, train_preds.shape)
        loss: torch.Tensor = loss_fn(train_preds, train_img_masks)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_sum += loss.item()

    model.eval()
    val_loss_sum = 0

    for val_batch_idx, (val_imgs, val_img_masks) in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            val_preds = model(val_imgs)
            loss = loss_fn(val_preds, val_img_masks)

        val_loss_sum += loss.item()
