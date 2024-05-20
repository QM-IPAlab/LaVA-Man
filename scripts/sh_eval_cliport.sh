export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae

exps_name="debug"
agent_name="mae_robot_lang"

# ======== task name ========= #

#task_name="put-block-in-bowl-unseen-colors"
#task_name="put-block-in-bowl-seen-colors"

#task_name="packing-seen-google-objects-group"
#task_name="packing-unseen-google-objects-group"

#task_name="packing-seen-google-objects-seq"
task_name="packing-unseen-google-objects-seq"


python cliport/vis_heatmap.py model_task=${task_name}\
                       eval_task=${task_name} \
                       agent=${agent_name} \
                       mode=test \
                       n_demos=100 \
                       train_demos=100 \
                       exp_folder=${exps_name} \
                       checkpoint_type=best \
                       update_results=True \
                       disp=False\
                       record.save_video=False