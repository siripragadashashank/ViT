import os
import torch
from lightning import ViT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets import ViTDataLoader


def train_model(**kwargs):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CHECKPOINT_PATH = "saved_models/new/"

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")]
                         )
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    vit_loader = ViTDataLoader(dataset_path='data')
    train_loader = vit_loader.get_train_loader()
    val_loader = vit_loader.get_val_loader()
    test_loader = vit_loader.get_test_loader()

    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")

    pretrained = os.path.join(CHECKPOINT_PATH, "ViT/lightning_logs/version_0/checkpoints/epoch=36-step=416250.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = ViT.load_from_checkpoint(pretrained_filename)
    elif os.path.isfile(pretrained):
        pl.seed_everything(42)
        # model = ViT(**kwargs)
        model = ViT.load_from_checkpoint(pretrained)
        trainer.fit(model, train_loader, val_loader)
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


if __name__ == '__main__':
    model, results = train_model(model_kwargs={
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'patch_size': 4,
        'num_channels': 3,
        'num_patches': 64,
        'num_classes': 10,
        'dropout': 0.2
    }, lr=3e-4)
    print("ViT results", results)
