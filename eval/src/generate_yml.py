import yaml

def generate_yaml(
    adapter_path: str, 
    base_path: str, 
):
    data = {
        "model": {
            "type": "base",
            "base_params": {
                "model_args": f"pretrained={adapter_path}",
                "dtype": "bfloat16"
            },
            "merged_weights": {
                "delta_weights": False,
                "adapter_weights": True,
                "base_model": base_path
            },
            "generation": {
                "multichoice_continuations_start_space": False,
                "no_multichoice_continuations_start_space": False
            }
        }
    }
    return yaml.dump(data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate a yml file')
    parser.add_argument(
        "--adapter_path",
        type=str
    )
    parser.add_argument(
        "--base_path",
        type=str
    )
    parser.add_argument(
        "--output_path",
        type=str
    )
    args = parser.parse_args()
    output_yml = generate_yaml(args.adapter_path, args.base_path)
    with open(args.output_path, "w") as f:
        f.write(output_yml)
