# ---------------------------------------------------------------------
# To run this script, run `python -m scripts.eval`
# Before running, edit the script to load desired model to evaluate
# ---------------------------------------------------------------------
import json
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2

from deeplab import inception_deeplabv3
from utils import ImageSegmentationDataset, BinaryMIOU


# ---------------------------------------------
# Configurations
# ---------------------------------------------

LOGS_PATH = "May08_16-34-26_ipv-desktop_1module-inception-deeplabv3"
CLEAN_DATA_DIR = "./data/model_training"
ONSITE_DATA_DIR = "./data/onsite"
with open(f"{LOGS_PATH}/results.json") as f:
    TRAINING_PARAMS = json.load(f)

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
model = inception_deeplabv3(
    backbone=TRAINING_PARAMS["backbone"],
    inception_module_count=TRAINING_PARAMS["inception_modules"],
)
model.load_state_dict(torch.load(f"{LOGS_PATH}/best_model_state_dict.pt"))
model.to(device)
model.eval()

miou_metric = BinaryMIOU(device=device)

# ---------------------------------------------
# Dataset preparation
# ---------------------------------------------
KSIZE = 5


# transformation for blurring images
def blur_transforms(ksize: int):
    return v2.Compose(
        [
            v2.Lambda(lambda x: torch.tensor(cv2.blur(x.numpy(), (ksize, ksize)))),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.to(device)),
        ]
    )


# transformation for images
transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.to(device)),
    ]
)


# create datasets
original_test_set = ImageSegmentationDataset(
    f"{CLEAN_DATA_DIR}/test/images",
    f"{CLEAN_DATA_DIR}/test/image_masks",
    transforms,
    transforms,
)
blurred_test_set = ImageSegmentationDataset(
    f"{CLEAN_DATA_DIR}/test/images",
    f"{CLEAN_DATA_DIR}/test/image_masks",
    blur_transforms(ksize=KSIZE),
    transforms,
)
onsite_dataset = ImageSegmentationDataset(
    f"{ONSITE_DATA_DIR}/images",
    f"{ONSITE_DATA_DIR}/image_masks",
    transforms,
    transforms,
)

# create dataloaders
dataloaders = {
    "original test set": DataLoader(original_test_set, batch_size=1),
    f"blurred clean test set (ksize {KSIZE})": DataLoader(
        blurred_test_set, batch_size=1
    ),
    "onsite dataset": DataLoader(onsite_dataset, batch_size=1),
}


# start evaluation
for dataset_name, dataloader in dataloaders.items():
    miou_sum = 0

    for idx, (imgs, masks) in enumerate(tqdm(dataloader, desc=dataset_name)):
        with torch.no_grad():
            preds = model(imgs)["out"]
            miou_sum += miou_metric(preds, masks)

    print(f"{dataset_name} mIoU: {miou_sum / (idx + 1)}")
