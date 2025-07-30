import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def apply_swapping(
    model_adapted: AutoModelForCausalLM, 
    model_instruct: AutoModelForCausalLM,
    swapping_indices: list[int] = [0, 1, -2, -1],
    **kwargs
) -> AutoModelForCausalLM:
    # Sanity check
    assert model_adapted.config.model_type == model_instruct.config.model_type, "The model types of the two models must be the same!"

    # Copy the first and last two layers from model_adapted to model_instruct
    print("\t[apply_swapping]: Copying the first and last two layers from model_adapted to model_instruct...")
    if model_adapted.config.model_type in ("llama", "qwen2"):
        layer_attr_names = [
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
        ]
    elif model_adapted.config.model_type == "gemma2":
        layer_attr_names = [
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm",    
        ]
    else:
        raise NotImplementedError
    for index in swapping_indices:
        print(f"\t[apply_swapping]: Now swapping weights for Layer {index}...")
        with torch.no_grad():
            for attr_name in layer_attr_names:
                if attr_name == "self_attn":
                    # Copy the weights for the self-attention layer
                    for attr_name2 in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                            getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight
                        )
                elif attr_name == "mlp":
                    # Copy the weights for the MLP layer
                    for attr_name2 in ["gate_proj", "up_proj", "down_proj"]:
                        getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                            getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight
                        )
                else:
                    # Copy the weights for the other layers
                    getattr(model_instruct.model.layers[index], attr_name).weight.copy_(
                        getattr(model_adapted.model.layers[index], attr_name).weight
                    )
    
    print("\t[apply_swapping]: Done!")
    return model_instruct