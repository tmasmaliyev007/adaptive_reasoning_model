from typing import Dict, List
from omegaconf import DictConfig

from transformers import PreTrainedTokenizerFast, PreTrainedModel
from trl import GRPOTrainer

def tokenize(
    example:   Dict[str, str],
    tokenizer: PreTrainedTokenizerFast,
    cfg:       DictConfig
) -> Dict[str, List[int]]:
    
    # Get Question
    question = example['question']


    # Define prompt message
    prompt_message = [
        {'role': 'user',      'content': question}
    ]

    # Tokenize only prompt message
    prompt_ids = tokenizer.apply_chat_template(
        prompt_message,
        tokenize = True,
        add_generation_prompt = True,
        truncation = True,
        max_length = cfg.model.max_prompt_length
    )

    # Return dictionary with corresponding fields
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels
    }

def prepare_grpo_dataset(ds: Dataset) -> Dataset:
    def process(example):
        return {
            'prompt': [
                {'role': 'user', 'content': example['question']}
            ],
            'answer': example['answer']
        }

    return ds.map(process, remove_columns=ds.column_names)

def push_to_hub_merged(
    trainer: GRPOTrainer,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DictConfig,
):
    # Unwrap the model
    unwrapped_model: PreTrainedModel = trainer.model

    # Merge LoRA weights into base model
    merged_model: PreTrainedModel = unwrapped_model.merge_and_unload()

    # Push merged model + tokenizer to Hub
    merged_model.push_to_hub(
        cfg.hub_repo,
        commit_message=f"Merged LoRA - {cfg.experiment_name}"
    )
    tokenizer.push_to_hub(
        cfg.hub_repo
    )