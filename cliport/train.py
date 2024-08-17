"""Main training script.

!!!! Warning: This script use Pytorch Lighting, but change the logger and original
lightning module. May need to TEST before submit to the github !!!!!

"""

from gc import callbacks
import os
from pathlib import Path
from re import T

import torch
from torch import max_pool1d
from torch.utils.data import random_split, DataLoader

from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from cliport.real_dataset import RealDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

WANDB_DIR = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/wandb_cliport'
TB_DIR = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/tensorboard'
os.environ["WANDB_SERVICE_WAIT"] = "60"

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    
    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    sep_mode = cfg['train']['sep_mode']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    batch_size = cfg['train']['batch_size']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Wandb Logger
    try:
        wandb_logger = WandbLogger(name=cfg['wandb']['run_name'],
                                tags=[f"{task}", f"{agent_type}", f"{sep_mode}"],
                                save_dir=WANDB_DIR,
                                mode="offline",
                                project='cliport') if cfg['train']['log'] else None
    except Exception as e :
        print("fail to initialize wandb. Continuing withour wandb")
        print(f"Error: {e}")
        cfg['train']['log'] = False
        wandb_logger = None
    
    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    callbacks=[]
    
    if sep_mode:
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg['wandb']['saver']['monitor'],
            filepath=os.path.join(checkpoint_path, f'{sep_mode}-best'),
            save_top_k=1,
            save_last=False,
        )

    else:
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg['wandb']['saver']['monitor'],
            filepath=os.path.join(checkpoint_path, 'best'),
            save_top_k=1,
            save_last=True,
        )

    # repeat the demos, to avoid too few steps per epoch
    n_cycle = (200//n_demos) if (batch_size!=1 and n_demos <= 100) else 1
    max_epochs = cfg['train']['n_steps'] // (cfg['train']['n_demos'] * n_cycle)
    
    acc = cfg['train']['accumulate_grad_batches'] if cfg['train']['accumulate_grad_batches'] else 1
    
    if cfg['train']['log']:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max_epochs // 20,
        resume_from_checkpoint=last_checkpoint,
        accumulate_grad_batches=acc,
        precision=cfg['train']['precision'],
        callbacks=callbacks
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    elif 'real' in dataset_type:
        train_ds = RealDataset(task_name=task, data_type='train', augment=True)
        val_ds = RealDataset(task_name=task,data_type='train', augment=False)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Set data loaders if batch_size > 1
    # When batch size > 1, return tensors, otherwise return numpy arrays as original
    if batch_size != 1:
        ###FIXME: The batchnorm=True is not saved in the hydra config file during training.
        ###       Be careful when evaluating the model.
        cfg['train']['batchnorm'] = True
        train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
        val_ds = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # Initialize agent
    if not sep_mode:
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
        if  cfg['train']['load_pretrained_ckpt']:
            pretrain_checkpoint = cfg['cliport_checkpoint']
            agent.load(pretrain_checkpoint)
            print('Loading from cliport_checkpoint: ', pretrain_checkpoint)

    else:
        # train the pick and place agents separately
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds, sep_mode)
    
    # Main training loop
    trainer.fit(agent)

if __name__ == '__main__':
    main()
