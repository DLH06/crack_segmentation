import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src import CrackDataset, UNet


class UNetLightning(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        max_epochs=50,
    ):
        super().__init__()
        self.model = UNet(in_channels=3, out_channels=1)  # Initialize your U-Net model
        self.loss_function = torch.nn.BCELoss()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

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
        total_loss = self.loss_function(outputs, targets)

        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        # Calculate the validation loss (if needed)
        val_loss = self.focal_loss_function(outputs, targets)

        # Log the validation loss and IoU for the batch
        self.log("val_loss", val_loss)  # Log the validation loss
        return val_loss


if __name__ == "__main__":
    # ============DATALOADER==============
    train_dataset = CrackDataset(root_dir="data/new_crack/train", image_size=448)
    test_dataset = CrackDataset(root_dir="data/new_crack/test", image_size=448)
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, num_workers=8, shuffle=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, num_workers=8, shuffle=False, pin_memory=True
    )
    # ====================================

    # =============CALLBACKS==============
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="{epoch}-{train_loss:.4f}-{val_loss:.4f}",  # Naming pattern for checkpoints
        save_top_k=-1,  # Number of best checkpoints to keep
        verbose=True,  # Print messages about checkpoint saving
    )
    # ====================================

    # =============LIGHTING===============
    model = UNetLightning(
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
