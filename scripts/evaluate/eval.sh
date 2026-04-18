#!/bin/bash

# Arguments
export DATASET=aime2025
export MODEL=qwen3-4B

export LOAD_NUM_OF_GPU_LAYERS=0
export SEQ_LENGTH_FOR_SLOT=4096
export CONCURRENT_CALLS=2

# Run python script to evaluate on the dataset
python -m evaluator                                              \
          model=${MODEL}                                         \
          dataset=${DATASET}                                     \
          load_num_of_gpu_layers=${LOAD_NUM_OF_GPU_LAYERS}       \
          concurrent_calls=${CONCURRENT_CALLS}                   \
