import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src import CrackDataset, DiceCrossEntropyLoss, DiceLoss, FocalLoss, IoU, UNet


class UNetLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=1.0,
        learning_rate=1e-3,
        max_epochs=50,
    ):
        super().__init__()
        self.model = UNet(in_channels=3, out_channels=1)  # Initialize your U-Net model
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.loss_function = DiceCrossEntropyLoss(
            ce_weight=self.ce_weight, dice_weight=self.dice_weight
        )
        self.focal_loss_function = FocalLoss(alpha=1, gamma=2)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.iou_metric = IoU(
            num_classes=num_classes
        )  # Create an instance of the IoU metric

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-8,
            betas=(0.9, 0.99),
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Adjust based on your validation metric
            },
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        # LOSS
        ce_loss = self.loss_function(outputs, targets)
        dice_loss = DiceLoss()(
            F.softmax(outputs, dim=1), F.one_hot(targets.long(), num_classes=outputs.shape[1])
        )
        focal_loss = self.focal_loss_function(outputs, targets)

        total_loss = (
            (self.ce_weight * ce_loss)
            + (self.dice_weight * dice_loss)
            + (self.focal_weight * focal_loss)
        )

        self.log("train_loss", total_loss)

        # IoU
        batch_iou = self.iou_metric.calculate_iou(
            F.softmax(outputs, dim=1), F.one_hot(targets.long(), num_classes=outputs.shape[1])
        )
        self.log("train_iou", batch_iou, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        # Calculate the validation loss (if needed)
        val_loss = self.loss_function(outputs, targets)
        
        # Calculate IoU for the current batch (if needed)
        batch_iou = self.iou_metric.calculate_iou(F.softmax(outputs, dim=1), F.one_hot(targets.long(), num_classes=outputs.shape[1]))

        # Log the validation loss and IoU for the batch
        self.log('val_loss', val_loss)  # Log the validation loss
        self.log('val_iou', batch_iou, on_step=False, on_epoch=True)  # Log IoU for the batch and aggregate over the epoch


if __name__ == "__main__":
    # ============DATALOADER==============
    train_dataset = CrackDataset(root_dir="data/train", image_size=448)
    test_dataset = CrackDataset(root_dir="data/test", image_size=448)
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, num_workers=8, shuffle=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=8, num_workers=8, shuffle=False, pin_memory=True
    )
    # ====================================

    # =============CALLBACKS==============
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="{epoch}-{val_loss:.4f}",  # Naming pattern for checkpoints
        monitor="val_loss",  # Metric to monitor (e.g., validation loss)
        save_top_k=3,  # Number of best checkpoints to keep
        mode="min",  # 'min' or 'max' depending on the monitored metric
        verbose=True,  # Print messages about checkpoint saving
    )
    # ====================================

    # =============LIGHTING===============
    model = UNetLightning(
        num_classes=2,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=1.0,
        learning_rate=1e-3,
        max_epochs=50,
    )
    # ====================================

    # ============TRAINING================
    trainer = pl.Trainer(
        max_epochs=50,
        logger=pl.loggers.TensorBoardLogger("logs/", name="unet_logs"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, test_dataloader)
