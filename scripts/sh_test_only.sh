export CLIPORT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/mae
export TOKENIZERS_PARALLELISM=false

agent_name="mae_fuse"
exps_name="exps_cliport/0425_fuse_multisze_m95_100demo"
train_demos=100
mae_model="mae_fuse"

#tasks for testing
tasks=("packing-seen-google-objects-group"\
    "packing-unseen-google-objects-group"\
    "packing-seen-google-objects-seq"\
    "packing-unseen-google-objects-seq"\
    "separating-piles-full"\
)
                         
for task in "${tasks[@]}"
do
    echo "Running evaluation for agent: $agent with task: $task"
    python cliport/eval_sep.py model_task=multi-language-conditioned-full\
                        eval_task=${task} \
                        agent=${agent_name} \
                        mode=test \
                        n_demos=100 \
                        train_demos=${train_demos} \
                        exp_folder=${exps_name} \
                        checkpoint_type=best\
                        update_results=True \
                        disp=True\
                        record.save_video=False

    python cliport/eval_sep.py model_task=multi-language-conditioned-full\
                        eval_task=${task} \
                        agent=${agent_name} \
                        mode=test \
                        n_demos=100 \
                        train_demos=${train_demos} \
                        exp_folder=${exps_name} \
                        checkpoint_type=last\
                        update_results=True \
                        disp=False\
                        record.save_video=False
done