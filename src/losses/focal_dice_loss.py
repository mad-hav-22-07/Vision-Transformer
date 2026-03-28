"""
Combined Focal + Dice Loss for Lane Segmentation.

Lane pixels are a small fraction of the total image (~5-15%), creating
severe class imbalance. This combined loss addresses this:

- Focal Loss: Down-weights well-classified (easy) pixels, focusing the
  model's learning on hard boundary pixels. Uses class-specific alpha
  weights to further compensate for imbalance.

- Dice Loss: Directly optimizes the overlap between predicted and ground
  truth regions per class, which is closely related to IoU/F1 metrics.

L_total = α × FocalLoss + β × DiceLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    where p_t is the model's estimated probability for the correct class.

    Args:
        alpha:       Per-class weight tensor of shape (num_classes,), or None for uniform.
        gamma:       Focusing parameter — higher values focus more on hard examples.
        reduction:   'mean' or 'sum'.
        ignore_index: Label index to ignore (e.g., 255 for unlabeled pixels).
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  Model output (B, C, H, W).
            targets: Ground truth class indices (B, H, W), values in [0, C-1].

        Returns:
            Scalar focal loss.
        """
        B, C, H, W = logits.shape

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # Flatten spatial dimensions
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # Filter out ignore_index
        valid_mask = targets_flat != self.ignore_index
        logits_flat = logits_flat[valid_mask]
        targets_flat = targets_flat[valid_mask]
        probs_flat = probs_flat[valid_mask]

        if targets_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Gather probability of the correct class: p_t
        p_t = probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Compute cross-entropy term: -log(p_t)
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Class-specific alpha weights
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets_flat]
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.

    Dice = 2 × |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Computed per-class and averaged. Smooth factor prevents division by zero
    for classes with no pixels.

    Args:
        smooth:       Smoothing factor to prevent division by zero.
        ignore_index: Label index to ignore.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  Model output (B, C, H, W).
            targets: Ground truth class indices (B, H, W).

        Returns:
            Scalar dice loss (mean across classes).
        """
        num_classes = logits.shape[1]

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_one_hot = F.one_hot(
            targets.clamp(0, num_classes - 1), num_classes
        ).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Mask out ignore_index pixels
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        # Flatten spatial dims: (B, C, H*W)
        probs = probs.flatten(2)
        targets_one_hot = targets_one_hot.flatten(2)

        # Dice score per class
        intersection = (probs * targets_one_hot).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average across classes and batch
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss.

    L_total = focal_weight × FocalLoss + dice_weight × DiceLoss

    Args:
        focal_weight: Weight for focal loss component.
        dice_weight:  Weight for dice loss component.
        alpha:        Per-class focal loss weights.
        gamma:        Focal loss focusing parameter.
        smooth:       Dice loss smoothing factor.
        ignore_index: Label index to ignore.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        smooth: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.focal = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            smooth=smooth,
            ignore_index=ignore_index,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            logits:  Model output (B, C, H, W).
            targets: Ground truth (B, H, W).

        Returns:
            Tuple of:
                - Total combined loss (scalar).
                - Dict with individual loss values for logging.
        """
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )

        loss_dict = {
            "focal_loss": focal_loss.item(),
            "dice_loss": dice_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict
