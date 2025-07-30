from transformers import AutoModelForCausalLM, AutoTokenizer
from methods import *

def main(args):
    # Load models
    print("Loading models...")
    model_src = AutoModelForCausalLM.from_pretrained(args.model_src_name_or_path, cache_dir=args.cache_dir)
    if args.model_src_base_name_or_path is not None:
        print("\tYou are loading CVA base models as well...")
        model_src_base = AutoModelForCausalLM.from_pretrained(args.model_src_base_name_or_path, cache_dir=args.cache_dir)
    else:
        model_src_base = None
    model_tgt = AutoModelForCausalLM.from_pretrained(args.model_tgt_name_or_path, cache_dir=args.cache_dir)

    # Load adapted tokenizer
    print("Loading tokenizer...")
    if args.model_src_base_name_or_path is not None:
        tokenizer_src = AutoTokenizer.from_pretrained(args.model_src_base_name_or_path, cache_dir=args.cache_dir)
    elif args.tokenizer_src_name_or_path is None:
        tokenizer_src = AutoTokenizer.from_pretrained(args.model_src_name_or_path, cache_dir=args.cache_dir)
    else:
        tokenizer_src = AutoTokenizer.from_pretrained(args.tokenizer_src_name_or_path, cache_dir=args.cache_dir)
    tokenizer_tgt = AutoTokenizer.from_pretrained(args.model_tgt_name_or_path, cache_dir=args.cache_dir)
    tokenizer_src.chat_template = tokenizer_tgt.chat_template

    # Merge models
    print("Merging models...")
    for func_name in args.pipeline:
        # get a function
        func = globals().get(func_name)
        # execute a method
        model_tgt = func(
            model_adapted=model_src, 
            model_adapted_base=model_src_base,
            model_instruct=model_tgt,
            tokenizer_adapted=tokenizer_src,
            tokenizer_instruct=tokenizer_tgt,
            **vars(args)
        )

    # Save merged model and tokenizer
    print("Saving merged model and tokenizer...")
    model_tgt.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size='8GB'
    )
    tokenizer_src.save_pretrained(args.output_dir)
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_src_name_or_path", type=str, required=True, help="The path to the CVA model or Merged model from mergekit.")
    parser.add_argument("--model_tgt_name_or_path", type=str, required=True, help="The path to the SFT model.")
    parser.add_argument("--model_src_base_name_or_path", type=str, help="The path to the CVA model.")
    parser.add_argument("--tokenizer_src_name_or_path", type=str, default=None, help="The path to the tokenizer of the CVA model.")
    parser.add_argument('--pipeline', type=str, nargs='+', required=True, default=None, help="Specify merging pipeline.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory to save the merged model and tokenizer.")
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache directory to save the downloaded models and tokenizers.")
    parser.add_argument("--consider_special_tokens", action="store_true", help="Whether to consider special tokens.")
    parser.add_argument("--consider_added_tokens", action="store_true", help="Whether to consider added tokens.")
    parser.add_argument("--copy_new_tokens_only", action="store_true", help="Whether to copy only new tokens.")
    parser.add_argument("--tie_weights", action="store_true", help="Whether to tie weights.")
    parser.add_argument("--swapping_indices", type=int, nargs='+', default=[0, 1, -2, -1], help="Specify the layer indices for layer swapping.")
    parser.add_argument("--transition_indices", type=int, nargs='+', default=[2, 3, -4, -3], help="Specify the layer indices for layer transition.")
    parser.add_argument("--transition_rates", type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5], help="Specify the transition rates for transition layers.")
    parser.add_argument("--transition_method", type=str, default="linear", choices=["linear", "slerp"], help="Specify the transition method for transition layers.")
    args = parser.parse_args()
    main(args)