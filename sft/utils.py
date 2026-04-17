from typing import Dict, List
from omegaconf import DictConfig

from transformers import PreTrainedTokenizerFast, PreTrainedModel
from trl import SFTTrainer
from accelerate import Accelerator

def tokenize(
    example:   Dict[str, str],
    tokenizer: PreTrainedTokenizerFast,
    cfg:       DictConfig
) -> Dict[str, List[int]]:
    
    # Get Instruction & Solution pairs
    instruction = example[cfg.data.user_field]
    solution    = example[cfg.data.assistant_field]

    # Define prompt message
    prompt_message = [
        {'role': 'user',      'content': instruction}
    ]

    # Tokenize only prompt message
    prompt_ids = tokenizer.apply_chat_template(
        prompt_message,
        tokenize = True,
        add_generation_prompt = True,
        truncation = True,
        max_length = cfg.model.max_seq_length
    )

    # Get length of tokenized prompt message
    prompt_length = len(prompt_ids)

    # Define full conversation
    full_message = [
        {'role': 'user',      'content': instruction},
        {'role': 'assistant', 'content': solution}
    ]

    # Tokenize full conversation
    input_ids = tokenizer.apply_chat_template(
        full_message,
        tokenize = True,
        add_generation_prompt = False,
        truncation = True,
        max_length = cfg.model.max_seq_length
    )

    # Set `ignore_index` to prompt tokens to ignore during loss calculation
    labels = [-100] * prompt_length + input_ids[prompt_length:]

    # Return dictionary with corresponding fields
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels
    }

def push_to_hub_merged(
    trainer: SFTTrainer,
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