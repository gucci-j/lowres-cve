import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
from peft import (LoraConfig, TaskType, get_peft_model)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          AutoConfig)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from util import CustomArgumentParser
from model import LlamaForMultiCausalLM


def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    if args.val_dataset_path is not None:
        val_dataset = datasets.load_from_disk(args.val_dataset_path)
    else:
        val_dataset = None

    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    #####
    # Set up the data collator
    #####
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #####
    # Load the model
    #####
    if args.is_baseline:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto", 
            cache_dir=args.cache_dir
        )
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        config.num_lm_heads = args.num_lm_heads
        if args.copy_lm_head:
            config.copy_lm_head = True
        else:
            config.copy_lm_head = False
        model = LlamaForMultiCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            cache_dir=args.cache_dir,
            config=config
        )
        if config.copy_lm_head:
            for i in range(config.num_lm_heads):
                with torch.no_grad():
                    model.lm_heads[i].weight.copy_(model.lm_head.weight)
    logger.info(model)

    # Set up LoRA
    if not args.no_lora:
        logger.info(f'Before PEFT applied (Memory): {model.get_memory_footprint()}')
    
        if args.model_type in ("llama2", "llama3", "gemma2"):
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                              "gate_proj", "down_proj", "up_proj"]
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")
        if args.tune_embeddings:
            if args.model_type in ("llama2", "llama3", "gemma2"):
                if args.is_baseline:
                    modules_to_save = ["lm_head", "embed_tokens"]
                else:
                    modules_to_save = ["lm_head", "embed_tokens", "lm_heads.0"]
            else:
                raise ValueError(f"Model type {args.model_type} not supported.")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                target_modules=target_modules,
                inference_mode=False, 
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
            )
            model.enable_input_require_grads() # This is important for PEFT to work: https://github.com/huggingface/peft/issues/1577
            model = get_peft_model(model, peft_config)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}) 

        logger.info(f'After PEFT applied (Memory): {model.get_memory_footprint()}')

    #####
    # Set up the trainer
    #####
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset
    )
    
    #####
    # Train the model
    #####
    trainer.train()

    #####
    # Save the model
    #####
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    logger.info(args)
    logger.info(training_args)

    main(args, training_args)
