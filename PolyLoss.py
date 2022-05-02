# from https://github.com/yiyixuxu/polyloss-pytorch --------------------------------------------------------------------

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = False,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: the shape should be BNH[WD] (one-hot format) or B1H[WD], where N is the number of classes.
                It should contain binary values

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            # BH[WD] -> BH1[WD]
            polyl = poly_loss.unsqueeze(1)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


# from https://github.com/frgfm/Holocron -------------------------------------------------------------------------------

def poly_loss2(
        x: Tensor,
        target: Tensor,
        eps: float = 2.,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
) -> Tensor:
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        x (torch.Tensor[N, K, ...]): predicted probability
        target (torch.Tensor[N, K, ...]): target probability
        eps (float, optional): epsilon 1 from the paper
        weight (torch.Tensor[K], optional): manual rescaling of each class
        ignore_index (int, optional): specifies target value that is ignored and do not contribute to gradient
        reduction (str, optional): reduction method
    Returns:
        torch.Tensor: loss reduced with `reduction` method
    """

    # log(P[class]) = log_softmax(score)[class]
    logpt = F.log_softmax(x, dim=1)

    # Compute pt and logpt only for target classes (the remaining will have a 0 coefficient)
    logpt = logpt.transpose(1, 0).flatten(1).gather(0, target.view(1, -1)).squeeze()
    # Ignore index (set loss contribution to 0)
    valid_idxs = torch.ones(target.view(-1).shape[0], dtype=torch.bool, device=x.device)
    if ignore_index >= 0 and ignore_index < x.shape[1]:
        valid_idxs[target.view(-1) == ignore_index] = False

    # Get P(class)
    loss = -1 * logpt + eps * (1 - logpt.exp())

    # Weight
    if weight is not None:
        # Tensor type
        if weight.type() != x.data.type():
            weight = weight.type_as(x.data)
        logpt = weight.gather(0, target.data.view(-1)) * logpt

    # Loss reduction
    if reduction == 'sum':
        loss = loss[valid_idxs].sum()
    elif reduction == 'mean':
        loss = loss[valid_idxs].mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)

    return loss


class PolyLoss2(_Loss):
    """Implements the Poly1 loss from `"PolyLoss: A Polynomial Expansion Perspective of Classification Loss
    Functions" <https://arxiv.org/pdf/2204.12511.pdf>`_.
    Args:
        weight (torch.Tensor[K], optional): class weight for loss computation
        eps (float, optional): epsilon 1 from the paper
        ignore_index: int = -100,
        reduction: str = 'mean',
    """

    def __init__(
            self,
            *args: Any,
            eps: float = 2.,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = eps

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return poly_loss2(x, target, self.eps, self.weight, self.ignore_index, self.reduction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self.eps}, reduction='{self.reduction}')"


if __name__ == '__main__':
    # Example of target in one-hot encoded format
    B, C, H, W = 16, 100, 3, 3
    input = torch.rand(B, C, H, W, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
    target_one_hot = to_one_hot(target[:, None, ...], num_classes=C)
    print(PolyLoss(softmax=True, reduction='mean')(input, target_one_hot))
    print(poly_loss2(input, target, eps=1.0))
    print(torch.nn.CrossEntropyLoss()(input, target))
    print(torch.nn.BCEWithLogitsLoss()(input, target_one_hot))


    # Example of target in one-hot encoded format
    B, C = 16, 100
    input = torch.rand(B, C, requires_grad=True)
    target = torch.randint(low=0, high=C - 1, size=(B,)).long()
    target_one_hot = to_one_hot(target[:, None, ...], num_classes=C)
    print(PolyLoss(softmax=True, reduction='mean')(input, target_one_hot))
    print(poly_loss2(input, target, eps=1.0))
    print(torch.nn.CrossEntropyLoss()(input, target))
    print(torch.nn.BCEWithLogitsLoss()(input, target_one_hot))

