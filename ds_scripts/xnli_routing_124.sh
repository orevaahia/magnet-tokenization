cat $0
echo "--------------------"

export PYTHONPATH=$(pwd)
export HF_HOME="cache"
export WANDB_CACHE_DIR="cache"

C=ds_configs/xnli_routing_124.yml
GPUS=4
config_file=configs/accelerate/gpu_4.yaml

SEEDS=(120)
LRS=(5e-5)
BSZS=(32)
LANGS=("es")


for SEED in "${SEEDS[@]}";do
    echo "Starting with seed ${SEED}"

    for LR in "${LRS[@]}";do

        for BSZ in "${BSZS[@]}";do

            for language in "${LANGS[@]}"; do
                work_dir="model_ckpts/downstream/xnli_joint_input_gridsearch_routing_1x_2x_4x/${language}_xnli_fixed_routing_seed${SEED}_bsz${BSZ}_lr${LR}_clip1.0_cosine_schedule"
                echo $work_dir
                echo 'Finding free port'
                PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
                accelerate launch --main_process_port=$PORT --config_file=$config_file --num_processes="$GPUS" src/train_classification.py  \
                    --config_file "$C" \
                    --work_dir $work_dir \
                    --language $language \
                    --lr  $LR \
                    --batch_size $BSZ \
                    --seed $SEED

            done
        done
    done
done

