import os
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from transformers import GemmaTokenizer

def main(args):
    # train a sentencepiece model on it
    # the settings here are (best effort) those used for training Llama 2
    options = dict(
        # input spec
        input=args.corpus_path, # path to the corpus
        input_format="text",
        # output spec
        model_prefix= args.output_dir + "target", # output filename prefix
        # algorithm spec
        # BPE alg
        model_type="bpe",
        vocab_size=args.vocab_size,
        # normalization
        normalization_rule_name="identity", # ew, turn off normalization
        remove_extra_whitespaces=False,
        input_sentence_size=200000000, # max number of training sentences
        max_sentence_length=4192, # max number of bytes per sentence
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.99995,
        byte_fallback=True,
        # merge rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0, # the UNK token MUST exist
        bos_id=1, # the others are optional, set to -1 to turn off
        eos_id=2,
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(), # use ~all system resources
    )
    spm.SentencePieceTrainer.train(**options)
    
    # load a model
    sp = spm.SentencePieceProcessor()
    sp.load(options["model_prefix"] + ".model")
    target_spm = sp_pb2_model.ModelProto()
    target_spm.ParseFromString(sp.serialized_model_proto())
    
    # load a original model
    base_sp_model = spm.SentencePieceProcessor()
    base_sp_model.Load(args.source_tokenizer_path)
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_sp_model.serialized_model_proto())
    base_spm_tokens_set=set(p.piece for p in base_spm.pieces)
    
    # merge a tokenizer
    print(len(base_spm_tokens_set))
    print(f"Before:{len(base_spm_tokens_set)}")
    added_pieces = []
    new_count = 0
    for p in target_spm.pieces:
        piece = p.piece
        if piece not in base_spm_tokens_set and new_count < args.num_new_tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            base_spm.pieces.append(new_p)
            added_pieces.append(piece)
            new_count += 1
    print(f"New model pieces: {len(base_spm.pieces)}")
    
    # save
    os.makedirs(args.output_dir,exist_ok=True)
    with open(args.output_dir +'/merged.model', 'wb') as f:
        f.write(base_spm.SerializeToString())
    tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
    tokenizer.save_pretrained(args.output_dir)

    # iteratively remove newly added but unused tokens untill all new tokens are used
    num_iter = 0
    while True:
        ## load the dataset
        dataset = load_dataset(
            "text", 
            data_files={"train": [args.corpus_path]},
            split="train"
        )

        ## tokenize the dataset
        dataset = dataset.map(
            lambda x: tokenizer(x["text"]),
            batched=True, remove_columns=dataset.column_names
        )

        ## get the token ids
        token_ids = set()
        for example in dataset:
            token_ids.update(example["input_ids"])

        ## remove the unused tokens
        num_removed = 0
        vocab = tokenizer.get_vocab()
        for p in base_spm.pieces:
            if vocab[p.piece] not in token_ids \
                and p.piece not in base_spm_tokens_set:
                base_spm.pieces.remove(p)
                num_removed += 1
        print(f"Removed {num_removed} unused tokens")
        if num_removed == 0 or num_iter > args.num_max_iter:
            ## save the updated model
            with open(args.output_dir +'/merged.model', 'wb') as f:
                f.write(base_spm.SerializeToString())
            tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
            tokenizer.save_pretrained(args.output_dir)
            break

        ## add the new tokens
        new_count = 0
        for p in target_spm.pieces:
            piece = p.piece
            if piece not in base_spm_tokens_set \
                and piece not in added_pieces \
                and new_count < num_removed:
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                base_spm.pieces.append(new_p)
                added_pieces.append(piece)
                new_count += 1
        print(f"New model pieces: {len(base_spm.pieces)}")

        ## save the updated model
        with open(args.output_dir +'/merged.model', 'wb') as f:
            f.write(base_spm.SerializeToString())
        tokenizer = GemmaTokenizer(vocab_file=args.output_dir+'/merged.model')
        tokenizer.save_pretrained(args.output_dir)

        num_iter += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path", 
        type=str,
        help="Path to the corpus to train the tokenizer on",
        required=True
    )
    parser.add_argument(
        "--vocab_size", 
        type=int,
        help="Vocabulary size of the tokenizer",
        required=True
    )
    parser.add_argument(
        "--source_tokenizer_path", 
        type=str,
        help="Path to the source tokenizer",
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Path to the output directory",
        required=True
    )
    parser.add_argument(
        "--lang_code", 
        type=str,
        help="Language code",
        required=True,
        choices=["ar", "ja", "de", "sw", "th", "hi", "el", "my", "si", "te"]
    )
    parser.add_argument(
        "--num_new_tokens", 
        type=int,
        help="Number of new tokens to add to the tokenizer",
        default=100
    )
    parser.add_argument(
        "--num_max_iter", 
        type=int,
        help="Maximum number of iterations to remove unused tokens",
        default=10
    )
    args = parser.parse_args()
    main(args)
    