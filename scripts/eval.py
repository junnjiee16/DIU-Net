# ---------------------------------------------------------------------
# To run this script, run `python -m scripts.eval`
# Before running, edit the script to load desired model to evaluate
#
# Things to evaluate:
# - Performance on normal test set
# - Performance on blurred test set (ksize of 3, 4, 5)
# - Performance on rotated test set
# ---------------------------------------------------------------------
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import v2

from diunet import DIUNet
from utils import ImageSegmentationDataset, BinaryMIOU


# ---------------------------------------------
# Configurations
# ---------------------------------------------
MODEL_PATH = "./logs/run_17-04-2024_15-51/best_model_state_dict.pt"
DATASET_DIR = "./data"
model = DIUNet(channel_scale=0.25, dense_block_depth_scale=0.25)

# check GPU availability
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(
        f"Info: CUDA GPU detected, using {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )
else:
    device = torch.device("cpu")
    print("Info: CUDA GPU not detected, using CPU")

# load model
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()


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
test_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/model_training/test/images",
    f"{DATASET_DIR}/model_training/test/image_masks",
    transforms,
    transforms,
)

rotated_test_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/rotated/test/images",
    f"{DATASET_DIR}/rotated/test/image_masks",
    transforms,
    transforms,
)

# create dataloaders
test_dataloader = DataLoader(test_dataset, batch_size=1)
rotated_test_dataloader = DataLoader(rotated_test_dataset, batch_size=1)


# ---------------------------------------------
# Evaluation
# ---------------------------------------------
miou_metric = BinaryMIOU(device=device)

# normal test set
miou_sum = 0
for idx, (imgs, masks) in enumerate(tqdm(test_dataloader, desc="Test set")):
    with torch.no_grad():
        preds = model(imgs)
        miou_sum += miou_metric(preds, masks)

print(f"Normal test set mIoU: {miou_sum / (idx + 1)}")

# rotated test set
rotated_miou_sum = 0
for idx, (imgs, masks) in enumerate(
    tqdm(rotated_test_dataloader, desc="Rotated test set")
):
    with torch.no_grad():
        preds = model(imgs)
        rotated_miou_sum += miou_metric(preds, masks)

print(f"Normal test set mIoU: {rotated_miou_sum / (idx + 1)}")