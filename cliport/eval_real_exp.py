"""
Scripts for evaluating the real-world experiments.
"""


import os
import json
import numpy as np
import hydra

#import sys
#os.system('export CLIPORT_ROOT=$(pwd)')
#import pdb; pdb.set_trace()
#sys.path.append(os.environ['CLIPORT_ROOT'])
# reason of previous error: the current root path is not in the python path
from cliport import agents
from cliport.utils import utils
from cliport.eval import list_ckpts_to_eval

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):

    # Load configs
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    tcfg['train']['exp_folder'] = vcfg['exp_folder']
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])
    eval_task = vcfg['eval_task']
    
    # load agent
    agent = agents.names[vcfg['agent']](name, tcfg, None, None)
    
    # Load checkpoint
    existing_results = {}
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)
    assert len(ckpts_to_eval) == 1, "Only one checkpoint should be evaluated at a time."
    ckpt = ckpts_to_eval[0]
    model_file = os.path.join(vcfg['model_path'], ckpt)

    agent.load(model_file)
    print(f"Loaded: {model_file}")

    # run prediction
    img = []  # make it [320, 160, 3]
    lang_goal = []

    # Attention model forward pass.
    pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
    pick_conf = agent.attn_forward(pick_inp)
    pick_conf = pick_conf.detach().cpu().numpy()
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0_pix = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

    # Transport model forward pass.
    place_inp = {'inp_img': img, 'p0': p0_pix, 'lang_goal': lang_goal}
    place_conf = agent.trans_forward(place_inp)
    place_conf = place_conf.permute(1, 2, 0)
    place_conf = place_conf.detach().cpu().numpy()
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

    # Pixels to end effector poses.
    pass

        


if __name__ == '__main__':
    main()
