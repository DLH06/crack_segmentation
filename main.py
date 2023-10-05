import argparse

import torch
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src import CrackDataset, FocalLoss, UNet


class UNetLightning(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        max_epochs=50,
    ):
        super().__init__()
        self.model = UNet(in_channels=3, out_channels=1)  # Initialize your U-Net model
        self.bce_function = torch.nn.BCELoss()
        self.focal_loss = FocalLoss()
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
        bce_loss = self.bce_function(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        total_loss = bce_loss + focal_loss

        self.log("bce_loss", bce_loss)
        self.log("focal_loss", focal_loss)
        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        # Calculate the validation loss (if needed)
        bce_loss = self.bce_function(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        val_loss = bce_loss + focal_loss

        # Log the validation loss and IoU for the batch
        self.log("val_loss", val_loss)  # Log the validation loss
        return val_loss


def cli():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_dir', type=str, required=True, help='Input training folder data')
    parser.add_argument('--val_dir', type=str, required=True, help='Input valid folder data')
    parser.add_argument('--output', type=str, default="checkpoints", help='output dir')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--image_size', type=int, default=448, help='image size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    parser.add_argument('--checkpoints', type=str, default=None, help='continue training from checkpoint')
    
    args = parser.parse_args()
    return args.__dict__


if __name__ == "__main__":
    opt = cli()

    # ============DATALOADER==============
    train_dataset = CrackDataset(root_dir=opt["train_dir"], image_size=opt["image_size"])
    test_dataset = CrackDataset(root_dir=opt["val_dir"], image_size=opt["image_size"])
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=False, pin_memory=True
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
        learning_rate=opt["learning_rate"],
        max_epochs=opt["max_epochs"],
    )
    if opt["checkpoints"] is not None:
        ckpt = torch.load(opt["checkpoints"])
        model.load_state_dict(ckpt["state_dict"])
    # ====================================

    # ============TRAINING================
    trainer = pl.Trainer(
        max_epochs=opt["max_epochs"],
        logger=pl.loggers.TensorBoardLogger("logs/", name="unet_logs"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_dataloader, test_dataloader)
