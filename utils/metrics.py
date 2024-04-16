import torch


def __calc_iou(bool_preds, bool_targets):
    intersection = torch.logical_and(bool_preds, bool_targets).sum().item()
    union = torch.logical_or(bool_preds, bool_targets).sum().item()

    if union == 0:
        return 0
    return intersection / union


def __calc_binary_miou(preds, targets):
    bool_preds1 = preds == 0
    bool_targets1 = targets == 0

    bool_preds2 = preds == 0
    bool_targets2 = targets == 0

    return (
        __calc_iou(bool_preds1, bool_targets1) + __calc_iou(bool_preds2, bool_targets2)
    ) / 2


def binary_miou(preds_batch, targets_batch):
    assert preds_batch == targets_batch

    iou_sum = 0
    for pred, target in zip(preds_batch, targets_batch):
        iou_sum += __calc_binary_miou(pred, target)

    return iou_sum / preds_batch.shape[0]
