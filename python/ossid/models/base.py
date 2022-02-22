import pytorch_lightning as pl
from omegaconf import OmegaConf

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = OmegaConf.to_container(config, resolve=True)
        self.save_hyperparameters()

        # OmegaConf.create(self.hparams)

        print(config)
