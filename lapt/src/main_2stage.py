import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          AutoConfig, get_cosine_schedule_with_warmup)

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
    # freeze the model
    for n, p in model.named_parameters():
        if "lm_head" not in n and "embed_tokens" not in n:
            p.requires_grad = False
    logger.info(model)

    #####
    # Set up the trainer
    #####
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
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
