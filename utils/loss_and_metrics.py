import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# New WeightedCELoss class
class WeightedCELoss(nn.Module):
    def __init__(self, num_classes, class_weights, sigma):
        super(WeightedCELoss, self).__init__()
        if num_classes != len(class_weights):
            raise ValueError("Number of classes and weights must match.")
        self.num_classes = num_classes
        self.class_weights = class_weights  # tensor on device
        self.sigma = sigma

    def forward(self, predictions, targets, distance_maps):
        # predictions: (N, C, H, W) logits
        # targets: (N, 1, H, W) class indices
        # distance_maps: (N, 1, H, W) distances to boundary

        epsilon = 1e-8
        position_weights = torch.exp(-(distance_maps**2) / (2 * (self.sigma**2 + epsilon)))

        # Compute CE loss per pixel (no reduction)
        ce_loss_per_pixel = F.cross_entropy(
            predictions, targets.squeeze(1),
            weight=self.class_weights,
            reduction='none'
        )  # shape (N, H, W)

        # Apply position weights
        weighted_ce_loss = ce_loss_per_pixel * position_weights.squeeze(1)

        return weighted_ce_loss.mean()

# # We finally didn't use DiceLoss
# # Modified DiceLoss without one-hot encoding to reduce memory.
# class DiceLoss(nn.Module):
#     # Original __init__ with weighting factors
#     # def __init__(self, num_classes=None, label_frequency=None, device=None):
#     def __init__(self, num_classes=None): # Modified: removed label_frequency and device arguments
#         super(DiceLoss, self).__init__()
#         self.num_classes = num_classes

#     def forward(self, pred, target):
#         # pred: (N, C, H, W) -> logits, target: (N, 1, H, W) -> class indices
#         pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
#         target = target.squeeze(1)  # (N, H, W)

#         N, C_pred, H, W = pred.shape
#         # Store dice for each sample and each class
#         dice_scores_per_sample_per_class = torch.zeros((N, self.num_classes), device=pred.device)

#         # Compute dice for each class (INCLUDING background 0)
#         for c in range(self.num_classes): # Modified: range includes 0 to average over all classes
#             pred_c = pred[:, c, :, :].contiguous().view(N, -1)  # (N, H*W) for class c
#             target_c = (target == c).float().contiguous().view(N, -1)  # (N, H*W) binary for class c

#             intersection = (pred_c * target_c).sum(dim=1)  # (N,)
#             union = pred_c.sum(dim=1) + target_c.sum(dim=1)  # (N,)

#             # Calculate Dice coefficient, adding a small epsilon to avoid division by zero
#             dice = (2. * intersection + 1e-8) / (union + 1e-8)
#             dice_scores_per_sample_per_class[:, c] = dice

#         # Calculate the mean dice score across all samples and all classes for simple averaging
#         mean_dice_score = dice_scores_per_sample_per_class.mean()

#         return 1 - mean_dice_score # Return 1 - mean_dice_score as the loss


def calculate_dice(pred, target, num_classes):
    """
    Calculate dice score components for each organ class.

    Returns:
        torch.Tensor: Shape (num_classes-1, 2) where for each organ (excluding background):
            [:, 0] = intersection count (only in slices where organ appears in ground truth)
            [:, 1] = total count (pred + target, only in slices where organ appears in ground truth)
    """
    # pred: (N, C, H, W), target: (N, 1, H, W)
    pred_labels = torch.argmax(pred, dim=1)  # (N, H, W)
    target = target.squeeze(1)  # (N, H, W)

    N = pred.shape[0]
    dice_components = torch.zeros(num_classes -1, 2, device=pred.device)

    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred_labels == c).float().view(N, -1)  # (N, H*W)
        target_c = (target == c).float().view(N, -1)  # (N, H*W)

        # Only consider slices where the organ appears in ground truth
        has_organ = target_c.sum(dim=1) > 0  # (N,)

        if has_organ.any():
            # Filter to only slices with this organ in ground truth
            pred_c_filtered = pred_c[has_organ]  # (N_filtered, H*W)
            target_c_filtered = target_c[has_organ]  # (N_filtered, H*W)

            intersection = (pred_c_filtered * target_c_filtered).sum()  # scalar
            total = pred_c_filtered.sum() + target_c_filtered.sum()  # scalar

            dice_components[c - 1, 0] = intersection
            dice_components[c - 1, 1] = total

    return dice_components


def calculate_iou(pred, target, num_classes):
    """
    Calculate IoU score components for each organ class.

    Returns:
        torch.Tensor: Shape (num_classes-1, 2) where for each organ (excluding background):
            [:, 0] = intersection count (only in slices where organ appears in ground truth)
            [:, 1] = union count (only in slices where organ appears in ground truth)
    """
    # pred: (N, C, H, W), target: (N, 1, H, W)
    pred_labels = torch.argmax(pred, dim=1)  # (N, H, W)
    target = target.squeeze(1)  # (N, H, W)

    N = pred.shape[0]
    iou_components = torch.zeros(num_classes - 1, 2, device=pred.device)

    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (pred_labels == c).view(N, -1)  # (N, H*W)
        target_c = (target == c).view(N, -1)  # (N, H*W)

        # Only consider slices where the organ appears in ground truth
        has_organ = target_c.sum(dim=1) > 0  # (N,)

        if has_organ.any():
            # Filter to only slices with this organ in ground truth
            pred_c_filtered = pred_c[has_organ]  # (N_filtered, H*W)
            target_c_filtered = target_c[has_organ]  # (N_filtered, H*W)

            intersection = (pred_c_filtered & target_c_filtered).float().sum()  # scalar
            union = (pred_c_filtered | target_c_filtered).float().sum()  # scalar

            iou_components[c - 1, 0] = intersection
            iou_components[c - 1, 1] = union

    return iou_components
