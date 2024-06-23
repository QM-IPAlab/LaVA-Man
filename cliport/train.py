"""Main training script."""

import os
from pathlib import Path

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

WANDB_DIR = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/wandb_cliport'

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg['wandb']['run_name'],
                               tags=cfg['wandb']['logger']['tags'],
                               offline=True,
                               save_dir=WANDB_DIR) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    acc = cfg['train']['accumulate_grad_batches'] if cfg['train']['accumulate_grad_batches'] else 1
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
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
    if last_checkpoint is None and cfg['cliport_checkpoint'] is not None:
        pretrain_checkpoint = cfg['cliport_checkpoint']
        agent.load(pretrain_checkpoint)
        print('Loading from cliport_checkpoint: ', pretrain_checkpoint)

    # Main training loop
    trainer.fit(agent)

if __name__ == '__main__':
    main()
