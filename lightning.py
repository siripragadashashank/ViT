import pytorch_lightning as pl


class LitVisionTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()