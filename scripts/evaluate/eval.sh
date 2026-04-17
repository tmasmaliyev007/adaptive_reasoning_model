export DATA_PATH=data/jsonl/AIME2025/test.jsonl
export EXPERIMENT_NAME=aime2025-qwen3-4B 
export CONCURRENT_CALLS=8

python -m evaluator \
          data_path=${DATA_PATH} \
          experiment_name=${EXPERIMENT_NAME} \
          concurrent_calls=${CONCURRENT_CALLS}