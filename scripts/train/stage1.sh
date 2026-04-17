export EXPERIMENT_NAME=test

accelerate launch --num_processes 2 \
                  -m sft.train \
                  experiment_name=$EXPERIMENT_NAME