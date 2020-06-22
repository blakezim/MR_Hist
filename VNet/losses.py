def dice_loss(pred, target, eps=0.0001):
    """ Dice Loss

    INPUT:
        pred: Prediction from the model - assumed 1 class
        target: Label or ground truth - assumed 1 class

    RETURN:
        ndice: (1 - Dice) for minimizing
    """

    pflat = pred.view(-1)
    tflat = target.view(-1)

    intersection = (pflat * tflat).sum()
    union = pflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice


