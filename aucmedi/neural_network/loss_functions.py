# Taken from https://github.com/itakurah/focal-loss-pytorch/blob/main/focal_loss.py and adapted
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", task_type="binary"):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.task_type = task_type
        if isinstance(alpha, (list, tuple)) or (
            hasattr(alpha, "__iter__") and not isinstance(alpha, torch.Tensor)
        ):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes) one-hot encoded
        """
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
            )

    def binary_focal_loss(self, inputs, targets):
        """Focal loss for binary classification."""
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification."""
        alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # Compute cross-entropy for each class
        ce_loss = -targets * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha (per-class weighting)
        if isinstance(alpha, torch.Tensor):
            target_idx = targets.argmax(dim=1).long().to(inputs.device)
            alpha_t = alpha.gather(0, target_idx)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss
        else:
            # TODO: Implement scalar alpha for multi-class if needed (not common, usually per-class weights are used)
            alpha_t

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        loss = loss.sum(dim=1)  # Sum over classes for each sample
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Focal loss for multi-label classification."""
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if isinstance(self.alpha, torch.Tensor):
            # Interpret alpha as class weights
            alpha_t = self.alpha.to(inputs.device)
            bce_loss = alpha_t * bce_loss
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        loss = loss.sum(dim=1)  # Sum over classes for each sample
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
