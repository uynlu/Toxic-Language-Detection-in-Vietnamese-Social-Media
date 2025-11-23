import argparse
import os
from tqdm import tqdm

from labeling.labeling_utils.sample import random_sample_from_json_files


parser = argparse.ArgumentParser(description="Sample data.")
parser.add_argument("--num-samples", type=int, required=True, default=10)
parser.add_argument("--input-folder", type=str, required=True)
parser.add_argument("--output-folder", type=str, required=True)
parser.add_argument("--sample-size", type=int, required=False, default=50)


if __name__ == "__main__":
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)

    for i in tqdm(range(1, args.num_samples + 1)):
        random_sample_from_json_files(
            input_folder=args.input_folder,
            output_file=os.path.join(args.output_folder, f"sample_{i}.json"),
            sample_size=args.sample_size
        )
