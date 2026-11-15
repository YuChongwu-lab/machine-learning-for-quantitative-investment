
import torch
import torch.nn as nn

class DirectionalLoss(nn.Module):
    """
    Directional loss: rewards correctly predicting the up/down direction.
    Idea:
    - if the predicted direction is correct, the loss is smaller
    - if the predicted direction is wrong, the loss is larger
    - combines the strengths of classification and regression
    """

    def __init__(self, alpha=1.0, beta=2.0, use_soft=True):
        """
        Args:
            alpha: weight when the direction is correct
            beta: weight when the direction is wrong (usually beta > alpha)
            use_soft: whether to use soft directional matching
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_soft = use_soft

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) predicted returns
            target: (B, T) true returns
        Returns:
            loss: scalar
        """
        if self.use_soft:
            # pred * target > 0 means the directions agree
            direction_agreement = pred * target
            # sigmoid maps it to [0, 1]
            direction_correct = torch.sigmoid(direction_agreement * 10)
        else:
            # original hard matching
            direction_correct = (torch.sign(pred) == torch.sign(target)).float()

        # base MSE loss
        mse = (pred - target) ** 2

        # lower the weight when the direction is correct, raise it when wrong
        # direction_correct near 1 -> weight near alpha
        # direction_correct near 0 -> weight near beta
        weights = self.alpha + (self.beta - self.alpha) * (1 - direction_correct)

        # weighted loss
        loss = (weights * mse).mean()

        return loss

