import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from datasets import ViTDataLoader
from models.cnn import CNN


class LiTCNN(pl.LightningModule):

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNN()
        vit_loader = ViTDataLoader(dataset_path="data")
        train_loader = vit_loader.get_train_loader()
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        preds = self.model(x)
        loss = F.cross_entropy(preds, y)
        acc = (preds.argmax(dim=-1) == y).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


