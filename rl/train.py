import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)    
from transformers import PreTrainedTokenizerFast, PreTrainedModel

from trl import GRPOConfig, GRPOTrainer

from .utils import push_to_hub_merged, tokenize, prepare_dataset
from .reward_fn import correctness_reward_func
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

    # Specify target modules
    target_modules = list(cfg.lora.target_modules)
    modules_to_save = []
    if cfg.lora.train_embeddings:
        modules_to_save.extend(["embed_tokens"])
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora.r,
        lora_alpha = cfg.lora.lora_alpha,
        lora_dropout = cfg.lora.lora_dropout,
        
        target_modules = target_modules,
        modules_to_save = modules_to_save,
        use_gradient_checkpointing = cfg.model.use_gradient_checkpointing
    )
    
    # Handle weight tying if needed
    if cfg.lora.tie_weights:
        model.base_model.model.tie_weights()

    return model, tokenizer


def load_local_dataset(cfg: DictConfig, tokenizer: PreTrainedTokenizerFast) -> Tuple[Dataset, Dataset]:
    # Read train & validation datasets from local directory
    ds_train = load_dataset("json", data_files={'train': cfg.data.train_path})['train']
    # ds_val   = load_dataset("json", data_files={'val':   cfg.data.eval_path})['val']

    # Prepare dataset on each example
    dst_train = prepare_dataset(ds_train)
    # dst_val   = prepare_dataset(ds_val)
    
    # dst_train = ds_train.map(
    #     lambda example: tokenize(example, tokenizer, cfg), 
    #     remove_columns=ds_train.column_names
    # )

    # dst_val = ds_val.map(
    #     lambda example: tokenize(example, tokenizer, cfg),
    #     remove_columns=ds_val.column_names
    # )

    return dst_train


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    dst_train: Dataset,
    # dst_val: Dataset,
    cfg: DictConfig
) -> GRPOTrainer:
    
    max_completion_length = cfg.model.max_seq_length - cfg.model.max_prompt_length

    # Define training argument
    args = GRPOConfig(
        # output_dir = f"checkpoints/{cfg.experiment_name}",
        num_train_epochs = cfg.training.num_epochs,
        beta = cfg.training.kl_coef,

        num_generations = cfg.training.num_generations,
        per_device_train_batch_size = cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.training.gradient_accumulation_steps, 

        max_prompt_length = cfg.model.max_prompt_length,
        max_completion_length = max_completion_length,

        warmup_ratio = cfg.training.warmup_ratio,
        learning_rate = cfg.training.learning_rate,
        lr_scheduler_type = cfg.training.lr_scheduler_type,
        max_grad_norm = cfg.training.max_grad_norm,

        fp16 = cfg.training.fp16,
        bf16 = cfg.training.bf16,

        logging_steps = cfg.training.logging_steps,
        eval_strategy = "no",
        eval_steps = cfg.training.eval_steps,

        load_best_model_at_end = cfg.training.load_best_model_at_end,
        greater_is_better = cfg.training.greater_is_better,

        optim = cfg.training.optim,
        weight_decay = cfg.training.weight_decay,
        seed = cfg.seed,

        # dataset_num_proc = cfg.training.dataset_num_proc,
        # packing=cfg.training.packing,
        

        report_to = "wandb" if cfg.wandb.enabled else "none",
        run_name = cfg.experiment_name
    )
    
    # Define trainer wrapper
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dst_train,
        # eval_dataset = dst_val,
        args = args,
        reward_funcs = [correctness_reward_func]
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
    dst_train = load_local_dataset(cfg, tokenizer)

    # Define Trainer wrapper & start training
    trainer = build_trainer(model, tokenizer, dst_train, cfg)
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