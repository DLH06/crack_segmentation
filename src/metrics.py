import torchmetrics
import torch


class IoU(torchmetrics.Metric):
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds, target):
        # Calculate IoU for a batch of predictions and targets
        intersection, union = self.calculate_iou(preds, target)
        self.intersection += intersection
        self.union += union

    def calculate_iou(self, preds, target):
        # Implement your IoU calculation here
        # preds and target should be tensors
        # Calculate intersection and union for each class
        intersection = torch.sum(preds * target, dim=(1, 2))
        union = torch.sum((preds + target) - (preds * target), dim=(1, 2))
        return intersection, union

    def compute(self):
        # Calculate IoU for all classes and then compute the mean IoU
        mean_iou = torch.mean(self.intersection / self.union)
        return mean_iou
