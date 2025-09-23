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
from cliport.test_susie import RavensDatasetSuSIE
from cliport.real_dataset import RealDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from cliport.real_dataset_207 import Real207Dataset
from cliport.real_dataset_ann import RealAnnDataset


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
    augment = cfg['dataset']['aug'] if 'aug' in cfg['dataset'] else True

    # Wandb Logger
    try:
        wandb_logger = WandbLogger(name=cfg['wandb']['run_name'],
                                tags=[f"{task}", f"{agent_type}", f"{sep_mode}"],
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
            monitor='vl/loss',
            dirpath=os.path.join(checkpoint_path),
            filename= f'{sep_mode}-best',
            save_top_k=1,
            save_last=False,
        )

    else:
        checkpoint_callback = ModelCheckpoint(
            monitor='vl/loss',
            dirpath=os.path.join(checkpoint_path),
            filename= f'best',
            save_top_k=1,
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    # repeat the demos, to avoid too few steps per epoch
    n_cycle = (200//n_demos) if (batch_size!=1 and n_demos <= 100) else 1
    max_epochs = cfg['train']['n_steps'] // (cfg['train']['n_demos'] * n_cycle)
    
    acc = cfg['train']['accumulate_grad_batches'] if cfg['train']['accumulate_grad_batches'] else 1
    
    if cfg['train']['log']:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    trainer = Trainer(
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        check_val_every_n_epoch=max_epochs // 20,
        accumulate_grad_batches=acc,
        precision=cfg['train']['precision'],
        devices = 1
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
    elif 'real' == dataset_type:
        train_ds = RealDataset(task_name=task, data_type='train', augment=True)
        val_ds = RealDataset(task_name=task,data_type='train', augment=False)
    elif 'real_all' == dataset_type:
        train_ds = RealDataset(task_name=task, data_type='train_all', augment=True)
        val_ds = RealDataset(task_name=task,data_type='train_all', augment=False)
    elif 'real_ann' == dataset_type:
        train_ds = RealAnnDataset(task_name=task, data_type="train_ann", augment=True)
        val_ds = RealAnnDataset(task_name=task, data_type='train_ann', augment=False)
    elif 'mix' == dataset_type:
        train_ds_sim = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds_sim = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
        
        train_ds_real_pack_obj = RealDataset(task_name="pack_objects", data_type='train', augment=True)
        val_ds_real_pack_obj = RealDataset(task_name="pack_objects",data_type='train', augment=False)

        train_ds_real_pick_b = RealDataset(task_name="blocks_in_bowl", data_type='train', augment=True)
        val_ds_real_pick_b = RealDataset(task_name="blocks_in_bowl",data_type='train', augment=False)
        
        train_ds = torch.utils.data.ConcatDataset([train_ds_sim, train_ds_real_pack_obj, train_ds_real_pick_b])
        val_ds = torch.utils.data.ConcatDataset([val_ds_sim, val_ds_real_pack_obj, val_ds_real_pick_b])
        print("Using mixed dataset")
    elif 'susie_sim' in dataset_type:
        train_ds = RavensDatasetSuSIE(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=augment)
        val_ds = RavensDatasetSuSIE(os.path.join(data_dir, '{}-test'.format(task)), cfg, n_demos=n_val, augment=False)
    elif 'susie_real' in dataset_type:
        train_ds = RealAnnDataset(task_name=task, data_type="train_ann", augment=True)
        val_ds = RealAnnDataset(task_name=task, data_type='train_ann', augment=False)
    elif 'mix_real' ==  dataset_type: 
        train_ds_real_pack_a = RealDataset(task_name="pack_objects", data_type='train_all', augment=True)
        val_ds_a = RealDataset(task_name="pack_objects", data_type='train_all', augment=False)
        
        train_ds_real_pick_b = RealAnnDataset(task_name="train_ann", data_type='train', augment=True)
        val_ds_b = RealAnnDataset(task_name="train_ann",data_type='train', augment=False)

        train_ds_real_pick_c = RealAnnDataset(task_name="train_ann2", data_type='train', augment=True)
        val_ds_c = RealAnnDataset(task_name="train_ann2", data_type='train', augment=False)
        
        train_ds = torch.utils.data.ConcatDataset([train_ds_real_pack_a, train_ds_real_pick_b, train_ds_real_pick_c])
        val_ds = torch.utils.data.ConcatDataset([val_ds_a, val_ds_b, val_ds_c])

    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=augment)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Set data loaders if batch_size > 1
    # When batch size > 1, return tensors, otherwise return numpy arrays as original
    if batch_size != 1:
        ###FIXME: The batchnorm=True is not saved in the hydra config file during training.
        ###       Be careful when evaluating the model.
        cfg['train']['batchnorm'] = True
        train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2)
        val_ds = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=False, num_workers=2)

    # Initialize agent
    if not sep_mode:
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds)
        if  cfg['train']['load_pretrained_ckpt']:
            pretrain_checkpoint = cfg['cliport_checkpoint']
            agent.load(pretrain_checkpoint)
            print('Loading from cliport_checkpoint: ', pretrain_checkpoint)

    else:
        agent = agents.names[agent_type](name, cfg, train_ds, val_ds, sep_mode)
        if  cfg['train']['load_pretrained_ckpt']:
            pretrain_checkpoint = cfg['cliport_checkpoint']
            agent.load(pretrain_checkpoint)
            print('Loading from cliport_checkpoint: ', pretrain_checkpoint)
        # train the pick and place agents separately
        
    # Main training loop
    trainer.fit(agent)
    #trainer.predict(agent)
    #tuner = Tuner(trainer)
    #tuner.lr_find(agent)

if __name__ == '__main__':
    main()
