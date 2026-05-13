import os
import random
from collections import defaultdict

from utils import load_json, save_json


def random_sample_from_folder(input_folder: str, output_file: str, sample_size: int = 30):
    """Randomly sample from each JSON file in the input folder and save to output file."""
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

def random_sample_from_json_file(input_file: str, output_file: str, sample_size: int = 30):
    """Randomly sample from a JSON file and save to output file."""
    data = load_json(input_file)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of items")
    
    data_by_category = defaultdict(list)
    for item in data:
        data_by_category[item["category"]].append(item)

    sampled_data = []
    for _, items in data_by_category.items():
        sample_size = min(sample_size, len(items))
        sample = random.sample(items, sample_size)
        sampled_data.extend(sample)

    save_json(sampled_data, output_file)
