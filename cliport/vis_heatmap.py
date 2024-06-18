"""
visulize intermediate features, affordance maps, attentions, etc.
"""
from visualizer import get_local

get_local.activate()

import torch.nn.functional as F
import cv2
import os
import json
import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment
import cv2
import torch

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
    # elif 'real' in dataset_type:
    #     ds = RealDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"), tcfg, n_demos=vcfg['n_demos'],
    #                      augment=False)
    # elif 'big' in dataset_type:
    #     ds = dataset.RavensBigDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
    #                                   tcfg,
    #                                   n_demos=vcfg['n_demos'],
    #                                   augment=False)
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

        results = []
        mean_reward = 0.0

        # Run testing for each training run (each checkpoint).
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)

            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']

            # Run testing and save total rewards with last transition info.
            # Each test is a full episode.
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

                # Each step in one episode.
                for idx in range(task.max_steps):  # 3
                    act = agent.act(obs, info, goal)
                    lang_goal = info['lang_goal']
                    print(f'Lang Goal: {lang_goal}')
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')

                    if done:
                        get_local.clear()
                        break

                    cache = get_local.cache

                    #image = cache['CLIPLingUNetLat.forward.img'][0]  # torch.Size([1, 3, 320, 320])
                    image = cache['MAESeg2Model.forward.x'][0]  # torch.Size([1, 3, 320, 320])
                    #image = vu.tensor_to_cv2_img(image, to_rgb=False)

                    #heatmap = cache['CLIPLingUNetLat.forward.out'][0]  # torch.Size([1, 3, 320, 320])
                    #heatmap = cache['MAESeg2Model.forward.x'][0]
                    #heatmap = heatmap.squeeze()

                    # os.makedirs(f'{save_path}/../vis', exist_ok=True)
                    # save = vu.save_tensor_with_heatmap(image, heatmap,
                    #                                    f'{save_path}/../vis/heatmap_video{i + 1:06d}_step{idx}.png',
                    #                                    l=lang_goal)
                    # input(f'save to {save_path}: {save}')
                    # get_local.clear()

                    # save samples
                    image = image.squeeze().transpose(1, 2, 0)
                    rgb = image[:,:,:3]
                    depth = image[:,:,3]

                    rgb = (rgb - rgb.min())/ (rgb.max() - rgb.min())
                    rgb = (rgb * 255).astype(np.uint8)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    depth_std = 0.00903967
                    depth_mean = 0.00509261
                    depth = np.uint8(((depth * depth_std) + depth_mean) * 255)
                    depth2 = np.uint8(((depth * depth_std) + depth_mean) * 255)
                    
                    depth2 = (depth2 - depth2.min())/ (depth2.max() - depth2.min())
                    depth2 = (depth2 * 255).astype(np.uint8)
                    cv2.imwrite('depth2.png', depth2)

                    cv2.imwrite('rgb.png', rgb)
                    cv2.imwrite('depth.png', depth)
                    input('save rgb and depth')

                results.append((total_reward, info))
                mean_reward = np.mean([r for r, i in results])
                print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

                if record:
                    env.end_rec()


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


def visulize_cross_attention(cross_attentions, image):
    # visulization cross attetnion
    for i, layer in enumerate(cross_attentions):
        layer = layer.permute(0, 2, 3, 1)
        values, _ = torch.sum(layer, dim=1, keepdim=True)
        # resize to 320,320
        layer = F.interpolate(values, size=(320, 320), mode='nearest', align_corners=False)
        layer = layer.squeeze()
        layer = layer.cpu().detach().numpy()

        save = vu.save_tensor_with_heatmap(image, layer, f'vis_feature{i}.png')

        # input(f'save {save}')


def visulize_infeats_kmeans(infeats, image, prompt, K=8, folder_name='vis', img_name='infeats'):
    """
    visulize infeats and save them
    Args:
        infeats (list of tensor): list of 7 with dimensions of
            [1,320,64,64]
            [1,640,32,32]
            [1,1280,32,32]
            [1,1280,32,32]
            [1,1280,64,64]
            [1,640,128,128]
            [1,320,128,128]
        image (tensor): image tensor
    """
    for i, infeat in enumerate(infeats):

        if len(infeat.shape) == 4:
            infeat = infeat.squeeze(0)

        if isinstance(infeat, torch.Tensor):
            infeat = infeat.cpu().detach().numpy()

        color_mask = vu.get_kmeans_features(infeat, K)

        is_saved = vu.save_feature_map(image, color_mask, folder_name, f'{img_name}_{i}', prompt=prompt)
        print(f'save {is_saved}')


def visulize_cross_kmeans(infeats, image, prompt, K=10):
    for i, cross in enumerate(infeats):
        cross = cross.squeeze()
        cross = cross.permute(2, 0, 1)
        cross = cross.cpu().detach().numpy()

        color_mask = vu.get_kmeans_features(cross, K)

        is_saved = vu.save_feature_map(image, color_mask, 'vis_cross', f'cross_{i}', prompt=prompt)
        # input(f'save {is_saved}')


if __name__ == '__main__':
    main()

# pca = PCA(n_components=3)
# transformed_data = pca.fit_transform(lat_fusion_np)

# min_val = transformed_data.min(0)
# max_val = transformed_data.max(0)
# color_mask = 255 * (transformed_data - min_val) / (max_val - min_val)
# color_mask = np.clip(color_mask, 0, 255).astype(np.uint8)
# color_mask = color_mask.reshape(height, width, 3)
# color_mask = cv2.resize(color_mask, (ori.shape[1], ori.shape[0]), interpolation=cv2.INTER_NEAREST)

# combined = cv2.addWeighted(ori, 0.7, color_mask, 0.3, 0)
