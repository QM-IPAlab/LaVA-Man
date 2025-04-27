"""
Perform the evaluation of pick and place sepearately
11-Jan-2024 by Chaoran Zhu
"""


import os
import pickle
import json

import numpy as np
import hydra
from sympy import N

#import sys
#os.system('export CLIPORT_ROOT=$(pwd)')
#import pdb; pdb.set_trace()
#sys.path.append(os.environ['CLIPORT_ROOT'])
# reason of previous error: the current root path is not in the python path
from pytorch_lightning import Trainer
from cliport import agents
from cliport import dataset
from cliport.real_dataset import RealDataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment
from cliport.real_dataset import RealDataset
from pytorch_lightning.loggers import WandbLogger
WANDB_DIR = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/wandb_cliport'
from cliport.real_dataset_207 import Real207Dataset

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):

    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    wandb_logger = WandbLogger(name=tcfg['wandb']['run_name'],
                               tags=tcfg['wandb']['logger']['tags'],
                               offline=True,
                               save_dir=WANDB_DIR) if tcfg['train']['log'] else None

    # Choose eval mode and task.
    mode = vcfg['mode']
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
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

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
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
        
            tcfg['train']['exp_folder'] = vcfg['exp_folder']
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)
            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")


            trainer = Trainer(
                logger=None
            )

            a = trainer.test(agent,ds)
        

def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            # modified
            print("[MODIFIED!!] No best val ckpt found. Using best.ckpt")
            ckpt = 'best.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == '__main__':
    main()
