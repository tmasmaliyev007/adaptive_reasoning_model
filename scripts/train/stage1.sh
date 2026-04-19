export MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=Qwen2.5-3B-Instruct-Perplexity-E3-BF16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sft.train \
          model.name=$MODEL_NAME \
          experiment_name=$EXPERIMENT_NAME