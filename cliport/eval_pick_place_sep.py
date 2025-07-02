"""Ravens main training script."""
import os
import json
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import hydra
from cliport import agents
from cliport import dataset
from cliport.eval_sep import CkptManager, get_model_path
from cliport.real_dataset import RealDataset
from cliport.utils import utils
from cliport.real_dataset import RealDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader
WANDB_DIR = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/wandb_cliport'
from cliport.real_dataset_207 import Real207Dataset
from cliport.real_dataset_ann import RealAnnDataset


@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):

    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    wandb_logger = WandbLogger(name=tcfg['wandb']['run_name'], # type: ignore
                               tags=tcfg['wandb']['logger']['tags'],
                               offline=True,
                               save_dir=WANDB_DIR) if tcfg['train']['log'] else None

    # Choose eval mode and task.
    mode = vcfg['mode']
    checkpoint_type = vcfg['checkpoint_type']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test', 'test_seen', 'test_unseen'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    elif dataset_type == 'real':
        ds = RealDataset(task_name=eval_task, data_type=mode, augment=False)
    elif dataset_type == 'real_ours':
        ds = Real207Dataset(task_name=eval_task, data_type=mode, augment=False)
    elif dataset_type == 'susie_real':
        ds = RealAnnDataset(task_name=eval_task, data_type=mode, augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)
    
    ds = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpt_manager = CkptManager(vcfg)
    if checkpoint_type == 'val_missing':
        eval_pick, eval_place = ckpt_manager.get_eval_list()
    elif checkpoint_type == 'last':
        eval_pick, eval_place = ckpt_manager.get_last_ckpt()
    else:
        eval_pick, eval_place = ckpt_manager.get_test_ckpt()
    #TODO: other checkpoint types
   
    tcfg['pretrain_path'] = None 
    tcfg['train']['batchnorm'] = True 
    agent = agents.names[vcfg['agent']](name, tcfg, None, ds, 'both')
    agent.eval()

    # Evaluation loop
    for ckpt_pick, ckpt_place in zip(eval_pick, eval_place):
        
        model_pick, model_place, test_name, to_eval = get_model_path(
            ckpt_pick, ckpt_place, vcfg, existing_results)
        
        if not to_eval:
            continue

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            
            # Load checkpoint
            agent.load_sep(model_pick, model_place)
            print(f"Loaded: {ckpt_pick}, {ckpt_place}")

            trainer = Trainer(
                logger=None
            )
        
            a = trainer.test(agent, ds)


if __name__ == '__main__':
    main()
