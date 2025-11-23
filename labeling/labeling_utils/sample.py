import os
import random

from utils import load_json, save_json


def random_sample_from_json_files(input_folder: str, output_file: str, sample_size: int = 30):
    """Randomly sample from each JSON file in the input folder and save to output file"""
    all_samples = [] 

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)

            data = load_json(file_path)

            if isinstance(data, list):
                sample_size = min(sample_size, len(data))
                sample = random.sample(data, sample_size)
                all_samples.extend(sample)

    save_json(all_samples, output_file)