import click
import subprocess
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME
import sys
import os
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
import datasets.datasets as datasets
import models.models as models

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=os.path.join(repo_path,'config/config.yaml'))
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
    set_deterministic()

    cfg = yaml.safe_load(open(config))
    # overwrite the data path in case we have defined in the env variables
    # if os.environ.get('TRAIN_DATABASE'):
    #     cfg['data']['data_dir'] = os.environ.get('TRAIN_DATABASE')

    #Load data and model
    if weights is None:
        model = models.DiffusionPoints(cfg)
    else:
        if test:
            # we load the current config file just to overwrite inference parameters to try different stuff during inference
            ckpt_cfg = yaml.safe_load(open(weights.split('checkpoints')[0] + '/hparams.yaml'))
            ckpt_cfg['train']['uncond_min_w'] = cfg['train']['uncond_min_w']
            ckpt_cfg['train']['uncond_max_w'] = cfg['train']['uncond_max_w']
            ckpt_cfg['train']['num_workers'] = cfg['train']['num_workers']
            ckpt_cfg['train']['n_gpus'] = cfg['train']['n_gpus']
            ckpt_cfg['train']['batch_size'] = cfg['train']['batch_size']
            ckpt_cfg['data']['num_points'] = cfg['data']['num_points']
            ckpt_cfg['data']['data_dir'] = cfg['data']['data_dir']
            ckpt_cfg['data']['gt_dir'] = cfg['data']['gt_dir']
            ckpt_cfg['diff']['s_steps'] = cfg['diff']['s_steps']
            ckpt_cfg['experiment']['id'] = cfg['experiment']['id']
            ckpt_cfg['experiment']['logdir'] = cfg['experiment']['logdir']

            if 'dataset_norm' not in ckpt_cfg['data'].keys():
                ckpt_cfg['data']['dataset_norm'] = False
                ckpt_cfg['data']['std_axis_norm'] = False
            if 'max_range' not in ckpt_cfg['data'].keys():
                ckpt_cfg['data']['max_range'] = 10.

            cfg = ckpt_cfg

        model = models.DiffusionPoints.load_from_checkpoint(weights, hparams=cfg)
        print(model.hparams)

    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                 dirpath=os.path.join(cfg['experiment']['logdir'], cfg['experiment']['id'], 'checkpoints'),
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1,
                                 save_last=True,
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
                          num_sanity_val_steps=0,
                          limit_val_batches=0.001,
                          accelerator='ddp',
                          )
    else:
        trainer = Trainer(gpus=cfg['train']['n_gpus'],
                          logger=tb_logger,
                          log_every_n_steps=100,
                          resume_from_checkpoint=checkpoint,
                          max_epochs= cfg['train']['max_epoch'],
                          callbacks=[lr_monitor, checkpoint_saver],
                          check_val_every_n_epoch=1,
                          num_sanity_val_steps=0,
                          limit_val_batches=0.001,
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
