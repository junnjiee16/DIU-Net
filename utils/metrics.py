import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex


class ModifiedBinaryJaccardIndex:
    def __init__(self, class_id=0, threshold=0.5, device=torch.device("cpu")):
        self.threshold = threshold
        self.class_id = class_id
        self.jaccard = BinaryJaccardIndex().to(device)

    def __call__(self, pred, real):
        assert pred.shape == real.shape

        # convert image into 0s and 1s only using simple threshold
        # flatten arrays into 1st dimension, keep the 0th dimension
        int_pred = torch.where(
            pred > self.threshold, torch.tensor(1), torch.tensor(0)
        ).flatten(1)
        int_real = torch.where(
            real > self.threshold, torch.tensor(1), torch.tensor(0)
        ).flatten(1)

        # iterate through all predictions
        results = np.empty((int_pred.shape[0]))

        for i in range(len(results)):
            # torchmetrics jaccard index function only calculates iou of
            # the truth labels
            bool_pred = int_pred[i] == self.class_id
            bool_real = int_real[i] == self.class_id

            iou = self.jaccard(bool_pred, bool_real)
            results[i] = float(iou) if not torch.isnan(iou) else 0

        return np.average(results)


class BinaryMIOU:
    def __init__(self, threshold=0.5, device=torch.device("cpu")):
        """
        If input are floats [0, 1], it will be set to 0s and 1s based on threshold.
        """
        self.threshold = threshold
        self.jaccard = BinaryJaccardIndex().to(device)

    def __call__(self, pred, real):
        assert pred.shape == real.shape

        # convert image into 0s and 1s only using simple threshold
        # flatten arrays into 1st dimension, keep the 0th dimension
        int_pred = torch.where(
            pred > self.threshold, torch.tensor(1), torch.tensor(0)
        ).flatten(1)
        int_real = torch.where(
            real > self.threshold, torch.tensor(1), torch.tensor(0)
        ).flatten(1)

        # iterate through all predictions
        results = np.empty((int_pred.shape[0]))

        for i in range(len(results)):
            # torchmetrics jaccard index function only calculates iou of
            # the truth labels (1s), hence labels need to be flipped
            # so that iou for both classes can be calculated respectively
            background_iou = self.jaccard(int_pred[i], int_real[i])

            # Assumes background is white (1) and target is black (0)
            flip_int_pred = int_pred[i] == 0
            flip_int_real = int_real[i] == 0
            target_iou = self.jaccard(flip_int_pred, flip_int_real)

            miou = (background_iou + target_iou) / 2
            results[i] = float(miou) if not torch.isnan(miou) else 0

        return np.average(results)
