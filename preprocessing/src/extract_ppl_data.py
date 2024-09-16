import random

# Function to extract lines from source file that are not in target file
def extract_unique_lines(
    source_file_path:str,
    target_file_path_1:str,
    target_file_path_2:str,
    output_file_path:str,
    num_lines:int=100000
):
    # Read target file and store lines in a set
    with open(target_file_path_1, 'r', encoding='utf-8') as target_file:
        target_lines = set(line.strip() for line in target_file)
    if target_file_path_2 is not None:
        with open(target_file_path_2, 'r', encoding='utf-8') as target_file:
            target_lines.update(line.strip() for line in target_file)
        
    # Read source file and filter out lines that are in target file
    unique_lines = []
    with open(source_file_path, 'r', encoding='utf-8') as source_file:
        for line in source_file:
            if line.strip() not in target_lines:
                unique_lines.append(line)
    
    # Randomly sample 100K lines
    if len(unique_lines) > num_lines:
        sampled_lines = random.sample(unique_lines, num_lines)
    else:
        sampled_lines = unique_lines

    # Write unique lines to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(sampled_lines)
        
    return 

def main(args):
    extract_unique_lines(
        args.source_file_path, args.target_file_path_1, args.target_file_path_2, args.output_file_path, args.num_lines
    )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file_path', type=str, required=True)
    parser.add_argument('--target_file_path_1', type=str, required=True)
    parser.add_argument('--target_file_path_2', type=str)
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('--num_lines', type=int, default=100000)
    args = parser.parse_args()
    
    main(args)
