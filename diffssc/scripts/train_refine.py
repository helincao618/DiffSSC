import click
import subprocess
import MinkowskiEngine as ME
import torch
import yaml
import sys
import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

import datasets.datasets_refine as datasets
import models.models_refine as models


@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=os.path.join(repo_path,'config/config_refine.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
@click.option('--test', '-t', is_flag=True, help='test mode')
def main(config, weights, checkpoint, test):
    cfg = yaml.safe_load(open(config))

    #Load data and model
    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    if weights is None:
        model = models.RefineDiffusion(cfg)
    else:
        model = models.RefineDiffusion.load_from_checkpoint(weights,hparams=cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                 dirpath=os.path.join(cfg['experiment']['logdir'], cfg['experiment']['id'], 'checkpoints'),
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1
                                 )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(cfg['experiment']['logdir'], cfg['experiment']['id']),
                                             default_hp_metric=False)

    #Setup trainer
    if torch.cuda.device_count() > 1:
        cfg['train']['n_gpus'] = torch.cuda.device_count()
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=1,
                          limit_val_batches=0.05,
                          accelerator='ddp',
                          #gradient_clip_val=0.5,
                          )
    else:
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=1,
                          limit_val_batches=0.05,
                          #gradient_clip_val=0.5,
                          )


    # Train!
    if test:
        print('TESTING MODE')
        trainer.test(model, data)
    else:
        print('TRAINING MODE')
        trainer.fit(model, data)

if __name__ == "__main__":
    main()
