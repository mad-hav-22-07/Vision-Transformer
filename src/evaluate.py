"""
Evaluation metrics for lane segmentation.

Computes standard segmentation metrics:
- mIoU (mean Intersection over Union) — primary metric
- Per-class IoU
- Pixel accuracy
- Precision, Recall, F1 per class
- Confusion matrix
"""

import numpy as np
import torch
from typing import Optional


class SegmentationMetrics:
    """
    Accumulates predictions over a dataset and computes segmentation metrics.

    Usage:
        metrics = SegmentationMetrics(num_classes=3)
        for images, masks in dataloader:
            preds = model(images).argmax(dim=1)
            metrics.update(preds, masks)
        results = metrics.compute()
    """

    def __init__(self, num_classes: int = 3, class_names: list[str] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        # Confusion matrix: shape (num_classes, num_classes)
        # confusion[pred][target] = count
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self):
        """Reset accumulated statistics."""
        self.confusion_matrix.fill(0)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ):
        """
        Accumulate a batch of predictions and targets.

        Args:
            predictions: Predicted class indices (B, H, W) — LongTensor.
            targets:     Ground truth class indices (B, H, W) — LongTensor.
            ignore_index: Label to ignore.
        """
        preds = predictions.detach().cpu().numpy().flatten()
        targs = targets.detach().cpu().numpy().flatten()

        # Filter out ignored pixels
        valid = targs != ignore_index
        preds = preds[valid]
        targs = targs[valid]

        # Accumulate confusion matrix
        for p, t in zip(preds, targs):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.confusion_matrix[p, t] += 1

    def compute(self) -> dict:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dict with:
                - 'miou': mean IoU across all classes
                - 'per_class_iou': dict of class_name → IoU
                - 'pixel_accuracy': overall pixel accuracy
                - 'per_class_precision': dict
                - 'per_class_recall': dict
                - 'per_class_f1': dict
                - 'confusion_matrix': numpy array
        """
        cm = self.confusion_matrix

        # Per-class metrics
        ious = {}
        precisions = {}
        recalls = {}
        f1s = {}

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[c, :].sum() - tp
            fn = cm[:, c].sum() - tp

            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-10)
            ious[self.class_names[c]] = float(iou)

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp + 1e-10)
            precisions[self.class_names[c]] = float(precision)

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn + 1e-10)
            recalls[self.class_names[c]] = float(recall)

            # F1 = 2 × Precision × Recall / (Precision + Recall)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1s[self.class_names[c]] = float(f1)

        # Mean IoU
        miou = float(np.mean(list(ious.values())))

        # Pixel accuracy = sum(diagonal) / sum(all)
        pixel_acc = float(np.trace(cm) / (cm.sum() + 1e-10))

        return {
            "miou": miou,
            "pixel_accuracy": pixel_acc,
            "per_class_iou": ious,
            "per_class_precision": precisions,
            "per_class_recall": recalls,
            "per_class_f1": f1s,
            "confusion_matrix": cm,
        }

    def print_report(self, results: dict | None = None):
        """Print a formatted metrics report."""
        if results is None:
            results = self.compute()

        print("\n" + "=" * 60)
        print("SEGMENTATION EVALUATION REPORT")
        print("=" * 60)

        print(f"\n  Overall Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"  Mean IoU (mIoU):        {results['miou']:.4f}")

        print(f"\n  {'Class':<15} {'IoU':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("  " + "-" * 47)

        for cls_name in self.class_names:
            iou = results['per_class_iou'][cls_name]
            prec = results['per_class_precision'][cls_name]
            rec = results['per_class_recall'][cls_name]
            f1 = results['per_class_f1'][cls_name]
            print(f"  {cls_name:<15} {iou:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

        print("\n" + "=" * 60)


def compute_batch_miou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> float:
    """
    Quick mIoU computation for a single batch (for training logging).

    Args:
        logits:  Model output (B, C, H, W).
        targets: Ground truth (B, H, W).
        num_classes: Number of classes.

    Returns:
        Mean IoU as a float.
    """
    preds = logits.argmax(dim=1)  # (B, H, W)

    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()

        if union > 0:
            ious.append((intersection / union).item())

    return float(np.mean(ious)) if ious else 0.0
