import json
import math
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


def create_mapping_dict(dict1, dict2):
    """
    This function takes two dictionaries with identical keys and corresponding offset mappings
    and returns a dictionary mapping input IDs from dict1 to their corresponding IDs in dict2,
    handling many-to-many relationships by creating lists of corresponding IDs and allowing
    for partial overlaps.
    
    Args:
        dict1 (dict): The first dictionary with 'input_ids' and 'offset_mapping' keys.
        dict2 (dict): The second dictionary with 'input_ids' and 'offset_mapping' keys.
    
    Returns:
        dict: A dictionary mapping input IDs from dict1 to lists of corresponding IDs in dict2.
    """
    mapping_dict = {}
    for i, (start, end) in enumerate(dict1['offset_mapping']):
        input_id = dict1['input_ids'][i]
        corresponding_ids = []
        for j, (start_j, end_j) in enumerate(dict2['offset_mapping']):
            # Check for any overlap within the ranges
            if (start <= start_j and end >= end_j) or (start_j <= start and end_j >= end):
                corresponding_ids.append(dict2['input_ids'][j])
        if corresponding_ids:
            mapping_dict[input_id] = corresponding_ids
    return mapping_dict


def _init_by_merge(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    source_head_embeddings: np.ndarray,
    target_head_embeddings: np.ndarray,
    tie_word_embeddings: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    # init
    target_model_state = json.loads(target_tokenizer.backend_tokenizer.model.__getstate__())
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    
    # get new vocab
    new_vocab = []
    for token in target_model_state["vocab"]:
        if token not in source_vocab:
            new_vocab.append(token)

    # identify new merges
    new_token_to_old_tokens = {}
    new_token_to_merge_indices = {}
    tmp_merges = []
    for index, merge in enumerate(target_model_state["merges"]):
        token_1, token_2 = merge.split(" ")
        token = token_1 + token_2
        if token in new_vocab:
            if token_1 not in new_vocab and token_2 not in new_vocab:
                # can map to existing tokens
                if new_token_to_merge_indices.get(token) is not None:
                    new_token_to_merge_indices[token].append(index)
                else:
                    new_token_to_merge_indices[token] = [index]
            else:
                # completely new
                tmp_merges.append((index, merge))

    # map new token to old tokens
    for token, indices in new_token_to_merge_indices.items():
        for index in indices:
            token_1, token_2 = target_model_state["merges"][index].split(" ")
            if new_token_to_old_tokens.get(token) is None:
                new_token_to_old_tokens[token] = [[source_vocab[token_1], source_vocab[token_2]]]
            else:
                new_token_to_old_tokens[token].append([source_vocab[token_1], source_vocab[token_2]])
    
    # process unprocessed merges
    def convert_merge_into_old_tokens(merge) -> bool:
        # Given a merge, return token indices
        token_1, token_2 = merge.split(" ")
        token = token_1 + token_2
        if token_1 in new_vocab and token_2 not in new_vocab: # -> new_merge + old
            if new_token_to_old_tokens.get(token_1) is not None: # -> have a map to existing tokens
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[new_token_to_old_tokens[token_1], source_vocab[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([new_token_to_old_tokens[token_1], source_vocab[token_2]])
                return True
            else: # -> have no map to existing tokens
                return False
        elif token_1 not in new_vocab and token_2 in new_vocab: # -> old + new_merge 
            if new_token_to_old_tokens.get(token_2) is not None: # -> have a map to existing tokens
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[source_vocab[token_1], new_token_to_old_tokens[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([source_vocab[token_1], new_token_to_old_tokens[token_2]])
                return True
            else: # -> have no map to existing tokens
                return False
        
        elif token_1 in new_vocab and token_2 in new_vocab: # -> new + new
            if new_token_to_old_tokens.get(token_1) is not None and new_token_to_old_tokens.get(token_2) is not None:
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[new_token_to_old_tokens[token_1], new_token_to_old_tokens[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([new_token_to_old_tokens[token_1], new_token_to_old_tokens[token_2]])
                return True
            else:
                return False
    
    while True:
        unmerged = []
        for index, merge in tmp_merges:
            if not convert_merge_into_old_tokens(merge):
                unmerged.append((index, merge))
        if len(unmerged) == 0:
            break
        if len(unmerged) == len(tmp_merges):
            # if there is only one merge and it is unmerged, then it is a completely new merge
            unmerged_index, unmerged_merge = unmerged[0]
            token_1, token_2 = unmerged_merge.split(" ")
            token = token_1 + token_2
            if token_1 in new_vocab and token_2 not in new_vocab: # -> new_merge + old
                new_token_to_old_tokens[token_1] = [source_tokenizer(token_1, add_special_tokens=False).input_ids]
            elif token_1 not in new_vocab and token_2 in new_vocab: # -> old + new_merge
                pass
            elif token_1 in new_vocab and token_2 in new_vocab: # -> new + new
                pass
            break
        else:
            tmp_merges = unmerged
    
    # expand the embeddings
    def compute_embeddings(indices) -> np.ndarray:
        if type(indices) == int:
            return source_embeddings[indices]
        else:
            embeds = []
            for index in indices:
                embeds.append(compute_embeddings(index))
            return np.mean(np.array(embeds), axis=0)
    
    for token, indices in new_token_to_old_tokens.items():
        embeds = []
        for index in indices:
            embeds.append(compute_embeddings(index))
        target_embeddings[
            target_vocab[token]
        ] = np.mean(np.array(embeds), axis=0)
    
    # init output projection
    if not tie_word_embeddings:
        print("You are using the output projection init.")
        def compute_head_embeddings(indices) -> np.ndarray:
            if type(indices) == int:
                return source_head_embeddings[indices]
            else:
                embeds = []
                for index in indices:
                    embeds.append(compute_embeddings(index))
                return np.mean(np.array(embeds), axis=0)

        for token, indices in new_token_to_old_tokens.items():
            embeds = []
            for index in indices:
                embeds.append(compute_head_embeddings(index))
            target_head_embeddings[
                target_vocab[token]
            ] = np.mean(np.array(embeds), axis=0)
            
        return target_embeddings, target_head_embeddings
    else:
        return target_embeddings, None


def _init_by_align(
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    target_embeddings: np.ndarray,
    target_head_embeddings: np.ndarray,
    dataset_path: str,
    consider_mean: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    # init
    overlapping_tokens = set(source_tokenizer.get_vocab().keys()) & set(target_tokenizer.get_vocab().keys())
    new_tokens = set(target_tokenizer.get_vocab().keys()) - overlapping_tokens
    new_vocab = {new_token: target_tokenizer.get_vocab()[new_token] for new_token in new_tokens}
    new_index_to_old_indices = {val: [] for val in new_vocab.values()}
    dataset = load_dataset(
        "text", 
        data_files={"train": [dataset_path]},
        split="train"
    )
    
    # calculate mapping
    for sample in dataset:
        text = sample['text']
        source = source_tokenizer(text, return_offsets_mapping=True)
        adapted = target_tokenizer(text, return_offsets_mapping=True)
        for new_index, old_indices in create_mapping_dict(adapted, source).items():
            if new_index in new_index_to_old_indices:
                new_index_to_old_indices[new_index].append(old_indices)
    
    # calculate frequency
    def calculate_frequency(index_map):
        frequency_map = {}
        for new_id, nested_lists in index_map.items():
            nested_tuples = tuple(tuple(inner_list) for inner_list in nested_lists)
            frequency_map[new_id] = Counter(nested_tuples)
        return frequency_map
    frequency_map = calculate_frequency(new_index_to_old_indices)

    if consider_mean:
        for i in range(len(source_tokenizer), len(target_tokenizer)):
            token = target_tokenizer.convert_ids_to_tokens(i)
            source_ids = source_tokenizer.convert_tokens_to_ids(source_tokenizer.tokenize(token))
            try:
                largest_key = max(frequency_map[i], key=frequency_map[i].get)
            except ValueError:
                frequency_map[i] = Counter({tuple(source_ids): 1})
                continue
            print("Before", frequency_map[i])
            frequency_map[i].update({tuple(source_ids): frequency_map[i][largest_key]})
            print("After", frequency_map[i])

    def normalize_frequencies(frequency_map):
        normalized_frequency_map = {}

        for new_id, frequency_counter in frequency_map.items():
            total_count = sum(frequency_counter.values())
            normalized_frequency_map[new_id] = {old_id: frequency / total_count for old_id, frequency in frequency_counter.items()}

        return normalized_frequency_map
    frequency_map = normalize_frequencies(frequency_map)
    
    # reinit embeddings
    for new_id, frequency_counter in frequency_map.items():
        if target_embeddings is not None:
            new_embedding = np.zeros(target_embeddings.shape[1])
            for old_id, frequency in frequency_counter.items():
                if len(old_id) == 1:
                    new_embedding += frequency * target_embeddings[old_id[0]]
                else:
                    new_embedding += frequency * np.mean(target_embeddings[list(old_id)], axis=0)
            if not frequency_counter == Counter():
                target_embeddings[new_id] = new_embedding
            
        if target_head_embeddings is not None:
            new_head_embedding = np.zeros(target_head_embeddings.shape[1])
            for old_id, frequency in frequency_counter.items():
                if len(old_id) == 1:
                    new_head_embedding += frequency * target_head_embeddings[old_id[0]]
                else:
                    new_head_embedding += frequency * np.mean(target_head_embeddings[list(old_id)], axis=0)
            if not frequency_counter == Counter():
                target_head_embeddings[new_id] = new_head_embedding
    
    return target_embeddings, target_head_embeddings


def instantiate_model_by_align(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    dataset_path: str = None,
    tie_word_embeddings: bool = False,
    use_only_merge_for_head: bool = False,
    use_only_merge_for_embeddings: bool = False,
    use_only_align: bool = False,
    consider_mean: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # init
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0), 
        np.std(source_embeddings, axis=0), 
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1])
    )
    target_embeddings[:source_embeddings.shape[0]] = source_embeddings
    if not tie_word_embeddings:
        source_head_embeddings = source_model.get_output_embeddings().weight.detach().numpy()
        target_head_embeddings = np.zeros(
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
            source_head_embeddings.shape[1])
        )
        target_head_embeddings[:source_head_embeddings.shape[0]] = source_head_embeddings
    else:
        source_head_embeddings = None
        target_head_embeddings = None
    
    # init by merge
    if not use_only_align:
        target_embeddings, target_head_embeddings = _init_by_merge(
            source_model, source_tokenizer, target_tokenizer, 
            source_embeddings, target_embeddings, 
            source_head_embeddings, target_head_embeddings, 
            tie_word_embeddings
        )
    
    # init by align
    if use_only_merge_for_head:
        target_embeddings, _ = _init_by_align(
            source_tokenizer, target_tokenizer, 
            target_embeddings, None, 
            dataset_path, consider_mean
        )
    elif use_only_merge_for_embeddings:
        _, target_head_embeddings = _init_by_align(
            source_tokenizer, target_tokenizer, 
            None, target_head_embeddings, 
            dataset_path, consider_mean
        )
    else:
        target_embeddings, target_head_embeddings = _init_by_align(
            source_tokenizer, target_tokenizer, 
            target_embeddings, target_head_embeddings, 
            dataset_path, consider_mean
        )
    
    # finalize
    target_model = source_model
    target_model.resize_token_embeddings(
        len(target_tokenizer), 
        pad_to_multiple_of=8 # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = torch.from_numpy(target_head_embeddings)
        
    return target_model, target_tokenizer
