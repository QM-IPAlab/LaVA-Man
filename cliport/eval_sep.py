"""Ravens main training script."""

from visualizer import get_local
#get_local.activate()

import os
import json

import numpy as np
import hydra
from regex import F
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
#from cliport.environments.environment import Environment
from cliport.environments.environment_ours import EnvironmentWhite as Environment
import cliport.utils.visual_utils as vu

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    checkpoint_type = vcfg['checkpoint_type']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
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
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
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

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            
            # Load checkpoint
            agent.load_sep(model_pick, model_place)
            print(f"Loaded: {ckpt_pick}, {ckpt_place}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']

            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos):
                print(f'Test: {i + 1}/{n_demos}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)

                # set task
                if 'multi' in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f'Evaluating on {task_name}')
                else:
                    task_name = vcfg['eval_task']
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(task)
                obs = env.reset()
                info = env.info
                reward = 0

                # Start recording video (NOTE: super slow)
                if record:
                    video_name = f'{task_name}-{i+1:06d}'
                    if 'multi' in vcfg['model_task']:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                for step in range(task.max_steps):
                    act = agent.act(obs, info, goal)
                    lang_goal = info['lang_goal']
                    print(f'Lang Goal: {lang_goal}')
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    # # # vis
                    # cache = get_local.cache
                    # image = cache['TransporterAgentSep.act.img'][-1]
                    # image = image[:,:,:3]
                    # image = image[:, :, ::-1]
                    # image=image.astype(np.uint8)

                    # heatmap = cache['TransporterAgentSep.act.place_heatmap'][-1]
                    # os.makedirs(f'{save_path}/../failure', exist_ok=True)
                    # save = vu.save_tensor_with_heatmap(image, heatmap,
                    #                                 f'{save_path}/../failure/heatmap_video{i + 1:06d}.png',
                    #                                 l=lang_goal)
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                    if done:
                        break


                results.append((total_reward))
                mean_reward = np.mean([r for r in results])
                print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {test_name}')

                # End recording video
                if record:
                    env.end_rec()

            all_results[test_name] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }

        # Save results in a json file.
        if vcfg['save_results']:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


class CkptManager():
    def __init__(self, vcfg):
        self.vcfg = vcfg
        self.pick_list = []
        self.place_list = []
        
        self.pick_last = []
        self.place_last = []

        self.pick_best = []
        self.place_best = []
        
        self._load_ckpts()

    def _load_ckpts(self):
        ckpts = os.listdir(self.vcfg['model_path'])
        ckpts = sorted(ckpts,reverse=True)
        
        def classify_ckpts(ckpts, ckpt_list, last, best):
            if 'best' in ckpts: best.append(ckpts)
            elif 'last' in ckpts: last.append(ckpts)
            else: ckpt_list.append(ckpts)
        
        for item in ckpts:
            if 'pick' in item:
                classify_ckpts(item, self.pick_list, self.pick_last, self.pick_best)
            elif 'place' in item:
                classify_ckpts(item, self.place_list, self.place_last, self.place_best)
            else:
                print(f"Unknown ckpt: {item}")
                continue
    
    def print_ckpts(self):
        print(f"Pick List: {self.pick_list}")
        print(f"Place List: {self.place_list}")
        print(f"Pick Last: {self.pick_last}")
        print(f"Place Last: {self.place_last}")
        print(f"Pick Best: {self.pick_best}")
        print(f"Place Best: {self.place_best}")

    def get_best_pick(self):
        return self.pick_best[0]
    
    def get_best_place(self):
        return self.place_best[-1]

    def get_last_pick(self):
        return self.pick_last[-1]
    
    def get_last_place(self):
        return self.place_last[0]
    
    def get_pick_list(self):
        return_list = []
        if self.pick_last is not None:
            return_list.append(self.get_last_pick())
        return_list.extend(self.pick_list)
        return return_list
    
    def get_place_list(self):
        return_list = []
        if self.place_last is not None:
            return_list.append(self.get_last_place())
        return_list.extend(self.place_list)
        return return_list

    def find_best_pick(self):
        """
        return: 
            - best pick ckpts list
            - best place ckpts list
        """
        # use place_best to find the best pick ckpt
        eval_place = self.get_best_place()
        eval_pick =  self.get_pick_list()
        eval_place = [eval_place] * len(eval_pick)

        return eval_pick, eval_place

    def find_best_place(self):
        """
        return:
            - best pick ckpts list
            - best place ckpts list
        """
        # use pick_best to find the best place ckpt
        eval_pick = self.get_best_pick()
        eval_place = self.get_place_list()
        eval_pick = [eval_pick] * len(eval_place)

        return eval_pick, eval_place

    def get_eval_list(self):
        eval_pick= []
        eval_place = []    
        # first use pick best and eval all place ckpts
        list_pick, list_place = self.find_best_pick()
        eval_pick.extend(list_pick)
        eval_place.extend(list_place)

        # then use place best and eval all pick ckpts
        list_pick, list_place = self.find_best_place()
        eval_pick.extend(list_pick)
        eval_place.extend(list_place)

        # then use best, best
        eval_pick.append(self.get_best_pick())
        eval_place.append(self.get_best_place())

        assert len(eval_pick) == len(eval_place)

        return eval_pick, eval_place

    def get_test_ckpt(self):
        result_jsons = [c for c in os.listdir(self.vcfg['results_path']) if "results-val" in c]
        if 'multi' in self.vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(self.vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            
            best_checkpoint_place = 'place-last.ckpt'
            best_success_pick = -1.0
            best_checkpoint_pick = 'pick-last.ckpt'
            best_success_place = -1.0
            
            for ckpt, res in eval_res.items():
                # find the best place ckpt
                # Be careful, place-best in ckpt means the ckpt is for testing pick !
                # Among all the pick ckpts, find the best place ckpt
                if 'place-best' in ckpt and res['mean_reward'] > best_success_pick:
                    best_checkpoint_pick = ckpt.split('+')[0]
                    best_success_pick = res['mean_reward']
                
                if 'pick-best' in ckpt and res['mean_reward'] > best_success_place:
                    best_checkpoint_place = ckpt.split('+')[1]
                    best_success_place = res['mean_reward']
                
            print("Find best checkpoint:", best_checkpoint_pick, best_checkpoint_place)
            return [best_checkpoint_pick], [best_checkpoint_place]        
        else:
            return [self.get_best_pick()], [self.get_best_place()]

    def get_last_ckpt(self):
        return [self.get_last_pick()], [self.get_last_place()]


def get_model_path(ckpt_pick, ckpt_place, vcfg, existing_results):
    model_pick = os.path.join(vcfg['model_path'], ckpt_pick)
    model_place = os.path.join(vcfg['model_path'], ckpt_place)

    test_name = f"{str(ckpt_pick)}+{str(ckpt_place)}"

    def file_exist(model_file):
        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            return False
        return True
    
    def results_not_exist(test_name, existing_results):
        """ if update results and the test name not in existing results 
        """  
        if not vcfg['update_results'] and test_name in existing_results:
            print(f"Skipping because of existing results for {test_name}.")
            return False
        return True 

    if file_exist(model_pick) and file_exist(model_place) and results_not_exist(test_name, existing_results):
        return model_pick, model_place, test_name, True
    else:
        return None, None, None, False


if __name__ == '__main__':
    main()
