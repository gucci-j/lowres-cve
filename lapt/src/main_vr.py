import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, Trainer, AutoConfig)

from util import CustomArgumentParser
from model import Gemma2ForMultiCausalLM

def main(args, training_args):
    #####
    # Load the dataset
    #####
    train_dataset = datasets.load_from_disk(args.dataset_path)
    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####
    # Set up the data collator
    #####
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    #####
    # Load the model
    #####
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    if args.model_type == "gemma2":
        config.num_lm_heads = 1
        model = Gemma2ForMultiCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            #attn_implementation="flash_attention_2",
            config=config
        )
        for i in range(config.num_lm_heads):
            with torch.no_grad():
                model.lm_heads[i].weight.copy_(model.lm_head.weight)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    for param in model.model.layers.parameters():
        param.requires_grad = False
    for index in [0, 1, -2, -1]:
        for param in model.model.layers[index].parameters():
            param.requires_grad = True
    logger.info(model)

    #####
    # Set up the trainer
    #####
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=None
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
