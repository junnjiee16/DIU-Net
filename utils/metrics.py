import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex


class BinaryMIOU:
    def __init__(self, threshold=0.5):
        """
        If input are floats [0, 1], it will be set to 0s and 1s based on threshold.
        """
        self.threshold = threshold
        self.jaccard = BinaryJaccardIndex()

    def __call__(self, pred, real):
        assert pred.shape == real.shape

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
            background_iou = self.jaccard(int_pred[i], int_real[i])

            # Assumes background is white (1) and target is black (0)
            flip_int_pred = int_pred[i] == 0
            flip_int_real = int_real[i] == 0
            target_iou = self.jaccard(flip_int_pred, flip_int_real)

            results[i] = float((background_iou + target_iou) / 2)

        return np.average(results)
