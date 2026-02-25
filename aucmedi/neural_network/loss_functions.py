# Taken from https://github.com/itakurah/focal-loss-pytorch/blob/main/focal_loss.py and adapted
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        """
        Focal Loss class for binary classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Focal loss for binary classification.
        :param inputs: Predictions (logits) from the model.
                       Shape: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape: (batch_size, num_classes) one-hot encoded
        """
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


class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        """
        Focal Loss class for multi-class classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple)) or (
            hasattr(alpha, "__iter__") and not isinstance(alpha, torch.Tensor)
        ):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """Focal loss for multi-class classification.
        :param inputs: Predictions (logits) from the model.
                       Shape: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape: (batch_size, num_classes) one-hot encoded
        """
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
            alpha_t = self.alpha
            ce_loss = alpha_t * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        loss = loss.sum(dim=1)  # Sum over classes for each sample
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLabelFocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        class_sparsity_coefficient=None,
        reduction="mean",
    ):
        """
        Focal Loss class for multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, should be a tensor for class-wise weights. If None, no class balancing is used.
        :param class_sparsity_coefficient: Optional factor that is multiplied with the loss for positive samples.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (list, tuple)) or (
            hasattr(alpha, "__iter__") and not isinstance(alpha, torch.Tensor)
        ):
            self.class_alphas = torch.Tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.class_alphas = alpha
        elif alpha is None:
            self.class_alphas = None
        else:
            raise ValueError(
                "class_weights must be a list, tuple, or tensor of class weights."
            )
        self.class_sparsity_coefficient = class_sparsity_coefficient

    def forward(self, inputs, targets):
        """Focal loss for multi-label classification.
        :param inputs: Predictions (logits) from the model.
                       Shape: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape: (batch_size, num_classes) one-hot encoded
        """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.class_alphas is not None:
            # TODO: Use pos_weight of BCEWithLogitsLoss instead
            weights = self.class_alphas.to(inputs.device)
            bce_loss = weights * bce_loss
        if self.class_sparsity_coefficient is not None:
            # Interpret as class_sparsity_coefficient
            sparse_t = self.class_sparsity_coefficient * targets + (1 - targets)
            bce_loss = sparse_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        # Sum over classes for each sample
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "sum_mean":
            loss = loss.sum(dim=1)
            return loss.mean()
        return loss
