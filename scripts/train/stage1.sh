export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=Qwen2.5-3B-Instruct-E3-BF16

python -m sft.train \
          model.name=$MODEL_NAME \
          experiment_name=$EXPERIMENT_NAME