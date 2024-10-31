cat $0
echo "--------------------"

export PYTHONPATH=$PWD

C=ds_configs/pawsx_routing_51020.yml
GPUS=2
config_file=configs/accelerate/gpu_2.yaml


SEEDS=(120)
LRS=(5e-5)
BSZS=(32 16)
LANGS=("es" "fr")


for SEED in "${SEEDS[@]}";do
    echo "Starting with seed ${SEED}"

    for LR in "${LRS[@]}";do

        for BSZ in "${BSZS[@]}";do

            for language in "${LANGS[@]}"; do
                work_dir="model_ckpts/downstream/pawsx_joint_input_gridsearch_routing_5x_10x_20x/${language}_pawsx_routing_seed${SEED}_bsz${BSZ}_lr${LR}_clip1.0_cosine_schedule"
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