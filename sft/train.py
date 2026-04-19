import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from transformers import PreTrainedTokenizerFast, PreTrainedModel
from unsloth import FastLanguageModel

from trl import SFTConfig
from .build_trainer import PerplexityLossSFTTrainer

from.utils import push_to_hub_merged, tokenize
from datasets import load_dataset, Dataset

import torch
import logging

from typing import Tuple

load_dotenv()
logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16" : torch.float16,
    "bfloat16": torch.bfloat16
}


def load_model_and_tokenizer(cfg: DictConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model.name,
        max_seq_length = cfg.model.max_seq_length,
        dtype = DTYPE_MAP[cfg.model.dtype],
        load_in_4bit = cfg.model.load_in_4bit,
        attn_implementation = cfg.model.attn_implementation
    )

    # Add ARM Special tokens
    if cfg.model.special_tokens:
        tokenizer.add_tokens(
            list(cfg.model.special_tokens), 
            special_tokens=True
        )

    # Specify target modules
    target_modules = list(cfg.lora.target_modules)
    if cfg.lora.train_embeddings:
        target_modules.extend(["embed_tokens"])
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora.r,
        lora_alpha = cfg.lora.lora_alpha,
        lora_dropout = cfg.lora.lora_dropout,
        target_modules = target_modules,
        use_gradient_checkpointing = cfg.model.use_gradient_checkpointing
    )
    
    # Handle weight tying if needed
    if cfg.lora.tie_weights:
        model.base_model.model.tie_weights()

    return model, tokenizer


def load_local_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizerFast) -> Tuple[Dataset, Dataset]:
    # Read train & validation datasets from local directory
    ds_train = load_dataset("json", data_files={'train': cfg.data.train_path})['train']
    ds_val   = load_dataset("json", data_files={'val':   cfg.data.eval_path})['val']

    # Apply tokenization on each example
    dst_train = ds_train.map(
        lambda example: tokenize(example, tokenizer, cfg), 
        remove_columns=ds_train.column_names
    )

    dst_val = ds_val.map(
        lambda example: tokenize(example, tokenizer, cfg),
        remove_columns=ds_val.column_names
    )

    return dst_train, dst_val


def build_trainer(
    model: PreTrainedModel, 
    dst_train: Dataset, 
    dst_val: Dataset, 
    cfg: DictConfig
):
    # Define training argument
    args = SFTConfig(
        # output_dir = f"checkpoints/{cfg.experiment_name}",
        num_train_epochs = cfg.training.num_epochs,
        per_device_train_batch_size = cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps, 

        max_length = cfg.model.max_seq_length,

        warmup_ratio = cfg.training.warmup_ratio,
        learning_rate = cfg.training.learning_rate,
        lr_scheduler_type = cfg.training.lr_scheduler_type,
        max_grad_norm = cfg.training.max_grad_norm,

        fp16 = cfg.training.fp16,
        bf16 = cfg.training.bf16,

        logging_steps = cfg.training.logging_steps,
        eval_strategy = "steps",
        eval_steps = cfg.training.eval_steps,

        load_best_model_at_end = cfg.training.load_best_model_at_end,
        greater_is_better = cfg.training.greater_is_better,

        optim = cfg.training.optim,
        weight_decay = cfg.training.weight_decay,
        seed = cfg.seed,

        dataset_num_proc = cfg.training.dataset_num_proc,
        packing=cfg.training.packing,

        report_to = "wandb" if cfg.wandb.enabled else "none",
        run_name = cfg.experiment_name
    )
    
    # Define trainer wrapper
    trainer = PerplexityLossSFTTrainer(
        model = model,
        train_dataset = dst_train,
        eval_dataset = dst_val,
        args = args
    )

    return trainer

@hydra.main(
    config_path="configs", 
    config_name="config", 
    version_base=None
)
def main(cfg: DictConfig):
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = OmegaConf.create(container)

    # W & B Initialize
    if cfg.wandb.enabled:
        import wandb

        wandb.init(
            project = cfg.wandb.project,
            name = cfg.experiment_name,
            config = container
        )
    
    # Load model & tokenizer based on given config
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Load datasets from local directory
    dst_train, dst_val = load_local_dataset(cfg, tokenizer)

    # Define Trainer wrapper & start training
    trainer = build_trainer(model, dst_train, dst_val, cfg)
    trainer.train()

    # Push to the huggingface hub as merged model
    if cfg.hub_push:
        push_to_hub_merged(trainer, tokenizer, cfg)

    # Shut Down W & A after training if enabled
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == '__main__':
    # Start the main process
    main()