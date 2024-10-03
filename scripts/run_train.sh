cat $0
echo "--------------------"

export PYTHONPATH=$(pwd)
export HF_HOME="cache"
export WANDB_CACHE_DIR="cache"

C=configs/routing_three_scripts_config/oscar_cyrl10x_latin5x_deva13x.yaml
GPUS=4
work_dir=model_ckpts/
accelerate_config_file=configs/accelerate/gpu_4.yaml

echo 'Run training...'

if [ -z $GPUS ]
then
    python src/train.py --config_file "$C" --work_dir $work_dir
else
    echo 'Finding free port'
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    accelerate launch --main_process_port=$PORT --config_file=$accelerate_config_file --num_processes="$GPUS" src/train.py --config_file "$C" --work_dir $work_dir
fi



