import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from ossid.models import getModel
from ossid.datasets import getDataloaders
from ossid.conf import postPrcoessConf

@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    print("------------ CONFIG ------------")
    print(OmegaConf.to_yaml(config))

    pl.seed_everything(42)

    cwd = hydra.utils.get_original_cwd()

    config = postPrcoessConf(config)
    
    # Resume training if resume_path is given
    if config.resume_path is not None:
        resume = True
        config.resume_path = os.path.join(cwd, config.resume_path)
        assert config.resume_version is not None, "resume_version should be defined if resume_path if already given"
        assert config.resume_id is not None, "resume_id should be defined if resume_path if already given"
        
        print("Resume training from (v %d, id %s): %s" % (config.resume_version, config.resume_id, config.resume_path))
    else:
        resume = False

    # Simply use the weights, but the training logs will be created separately
    if config.weights_path is None:
        config.weights_path = config.resume_path
    else:
        config.weights_path = os.path.join(cwd, config.weights_path)

    # Handle the logging part
    config.exp_name = "_".join([config.model.name, config.dataset.name, config.exp_suffix])

    log_folder = os.path.join(cwd, 'lightning_logs', config.exp_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)


    # logger = pl_loggers.TensorBoardLogger(
    #     os.path.join(cwd, 'lightning_logs'), name=config.exp_name
    # )

    # Get the version number
    if not resume:
        v_num = 0
        while os.path.exists(os.path.join(log_folder, "config_v%d.yaml" % v_num)):
            v_num += 1
    else:
        v_num = config.resume_version
    config.exp_name = "%s_v%d" % (config.exp_name, v_num)

    print("Experiment name:", config.exp_name)
    

    # Get the model and dataloaders
    train_loader, valid_loader, test_loader = getDataloaders(config)
    model = getModel(config)

    # Initialize the logger and 
    wandb_folder = os.path.join(log_folder, "wandb_v%d" % v_num)
    if not os.path.exists(wandb_folder):
        os.makedirs(wandb_folder)
    logger = pl_loggers.WandbLogger(
        name=config.exp_name, project=config.project, save_dir=wandb_folder, id=config.resume_id
    )

    # log and also save hyperparamters
    logger.log_hyperparams(config)
    OmegaConf.save(config=config, f=os.path.join(log_folder, "config_v%d.yaml" % v_num))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # filepath = os.path.join(log_folder, "ckpts_v%d" % v_num, "{epoch:03d}-{epoch_train_loss:.3f}-{epoch_valseen_loss:.3f}-{epoch_valunseen_loss:.3f}"),
        dirpath = os.path.join(log_folder, "ckpts_v%d" % v_num),
        filename = config.model.checkpoint_name,
        monitor= config.model.monitor,
        mode = config.model.monitor_mode,
        save_last = True,
        save_top_k = config.model.save_top_k,
    )

    trainer = pl.Trainer(
        default_root_dir=cwd,
        gpus = config.train.gpus,
        accelerator = 'ddp',
        resume_from_checkpoint = config.resume_path,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=config.model.max_epochs,
        deterministic=True,
    )

    trainer.fit(model, train_loader, valid_loader)

    if type(test_loader) is list:
        for i_loader, loader in enumerate(test_loader):
            print("Test loader No:", i_loader)
            trainer.test(
                model, loader, 
                ckpt_path='best'
            )
    else:
        trainer.test(
            model, test_loader,
            ckpt_path='best'
        )

if __name__ == "__main__":
    main()
