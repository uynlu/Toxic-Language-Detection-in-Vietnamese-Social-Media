import os
import pandas as pd
import argparse

from utils import load_json, save_json
from preprocessing.preprocessing_utils import (
    process_youtube_comments,
    process_tiktok_comments,
    process_facebook_comments,
    process_reddit_comments,
    add_id,
    remove_urls,
    remove_duplicates,
    remove_hashtags,
    to_lowercase,
    clean_whitespace,
    filter_non_vietnamese_texts
)

parser = argparse.ArgumentParser(description="Preprocess data.")
parser.add_argument("--raw-folder", type=str, required=True)
parser.add_argument("--preprocessed-folder", type=str, required=True)
parser.add_argument("--flag", default=False, type=bool, required=False)


def preprocess_platform(target_category, platform, raw_folder, preprocessed_folder):
    """Preprocess data for a specific platform and target category."""
    input_folder = os.path.join(raw_folder, target_category, platform)

    output_folder = os.path.join(preprocessed_folder, "original")
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        data = pd.read_csv(os.path.join(input_folder, file_name))
        if platform == "tiktok":
            data = process_tiktok_comments(data)
        elif platform == "youtube":
            data = process_youtube_comments(data)
        elif platform == "reddit":
            data = process_reddit_comments(data)
        elif platform == "facebook":
            data = process_facebook_comments(data)
        else:
            raise ValueError("This platform is not supported.")
        
        comments = []
        for comment in data:
            comments.append({
                "text": comment,
                "category": target_category
            })
        
        output_path = os.path.join(output_folder, f"original_{target_category}.json")
        if os.path.exists(output_path):
            existing_data = load_json(output_path)
        else:
            existing_data = []

        existing_data.extend(comments)
        save_json(existing_data, output_path)

def preprocess(file_name, raw_folder, preprocessed_folder):
    data = load_json(os.path.join(raw_folder, file_name))

    file_name = file_name.split("_")[1]
    
    def save_file(file_name, task, preprocessed_folder):
        output_path = os.path.join(preprocessed_folder, f"{task}")
        os.makedirs(output_path, exist_ok=True)

        base, ext = os.path.splitext(file_name)
        processed_file_name = base + f"_{'_'.join(task.split(' '))}" + ext
        processed_file_path = os.path.join(output_path, processed_file_name)

        save_json(data, processed_file_path)

    if os.path.exists(os.path.join(preprocessed_folder, "filtered non vietnamese texts", f"{file_name.split('.')[0]}_logs")) == False:
        data = add_id(data)
        save_file(file_name, "added id", preprocessed_folder)
        data = remove_duplicates(data)
        save_file(file_name, "removed duplicates", preprocessed_folder)
        data = remove_urls(data)
        save_file(file_name, "removed urls", preprocessed_folder)
        data = remove_hashtags(data)
        save_file(file_name, "removed hashtags", preprocessed_folder)
        data = clean_whitespace(data)
        save_file(file_name, "cleaned whitespace", preprocessed_folder)
        data = to_lowercase(data)
        save_file(file_name, "lowercased", preprocessed_folder)
        data = filter_non_vietnamese_texts(data, os.path.join(preprocessed_folder, "filtered non vietnamese texts", f"{os.path.splitext(os.path.basename(file_name))[0]}_logs"))
        save_file(file_name, "filtered non vietnamese texts", preprocessed_folder)
    else:
        data = filter_non_vietnamese_texts(data, os.path.join(preprocessed_folder, "filtered non vietnamese texts", f"{os.path.splitext(os.path.basename(file_name))[0]}_logs"), flag=True)
        save_file(file_name, "filtered non vietnamese texts", preprocessed_folder)
    

if __name__ == "__main__":
    args = parser.parse_args()

    if args.flag:
        for target_category in os.listdir(args.raw_folder):
            if "link" not in target_category:
                print(f"Processing {target_category}!")
                for platform in os.listdir(os.path.join(args.raw_folder, target_category)):
                    preprocess_platform(
                        target_category=target_category,
                        platform=platform,
                        raw_folder=args.raw_folder,
                        preprocessed_folder=args.preprocessed_folder
                    )
    else:
        for file_name in os.listdir(args.raw_folder):
            print(f"Processing {file_name}!")
            preprocess(
                file_name=file_name,
                raw_folder=args.raw_folder,
                preprocessed_folder=args.preprocessed_folder
            )
