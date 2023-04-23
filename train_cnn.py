import os
import torch
import pytorch_lightning as pl
from datasets import ViTDataLoader
from lightning_cnn import LiTCNN
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def train_model(batch_size=128, **kwargs):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CHECKPOINT_PATH = "saved_models/new/"

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "CNN"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")]
                         )

    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    vit_loader = ViTDataLoader(dataset_path='data', batch_size=batch_size)
    train_loader = vit_loader.get_train_loader()
    val_loader = vit_loader.get_val_loader()
    test_loader = vit_loader.get_test_loader()

    pl.seed_everything(42)
    model = LiTCNN(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = LiTCNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == '__main__':
    cnn_model, results = train_model(lr=3e-4)
    print("CNN results", results)
