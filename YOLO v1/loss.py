import torch
import torch.nn as nn
from utils import intersection_over_union

class Yolov1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(Yolov1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S: split size of the image
        B: number of boxes
        C: number of classes
        """
        self.S = S
        self.B = B
        self.C = C

        # these parameters are determined in the paper
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S * S * (C + B*5))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # IoU is implemented in utils
        iou1 = intersection_over_union(predictions[..., 21: 25], target[..., 21: 25])
        iou2 = intersection_over_union(predictions[..., 26: 30], target[..., 21: 25])
        ious = torch.cat(iou1.unsqueeze(0), iou2.unsqueeze(0), dim=0)

        # Take the box with highest IoU out of the two prediction
        # bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)

        exist_box = target[..., 20].unsqueeze(3)  # Iobj_i