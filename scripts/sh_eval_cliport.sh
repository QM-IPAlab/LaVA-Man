# #!/bin/bash
# #SBATCH --partition=devel
# #SBATCH --gres=gpu:1
# #SBATCH --job-name=failure
# #SBATCH --cpus-per-task=16
# module load python/anaconda3
# source activate mae-cliport
export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="exps_real_ann_sep_seg2_add"
agent_name="mae_sep_seg2_add"

# ======== task name ========= #

task_name="pack_objects"
#task_name="put-block-in-bowl-seen-colors"

#task_name="packing-seen-google-objects-group"
#task_name="packing-unseen-google-objects-group"

#task_name="packing-seen-google-objects-seq"
#task_name="packing-unseen-google-objects-seq"

python cliport/eval_pick_place_sep.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=last \
                       update_results=True \
                       disp=False\
                       record.save_video=False

# python cliport/vis_failure.py model_task=multi-language-conditioned\
#                        eval_task=${task_name} \
#                        agent=${agent_name} \
#                        mode=test \
#                        n_demos=100 \
#                        train_demos=1000 \
#                        exp_folder=${exps_name} \
#                        checkpoint_type=test_best \
#                        update_results=True \
#                        disp=False\
#                        record.save_video=True


