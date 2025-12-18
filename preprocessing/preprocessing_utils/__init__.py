import re
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from openai import OpenAI

from utils import save_json, load_json


load_dotenv()
API_KEY = os.getenv("GPT_API_KEY")


def process_tiktok_comments(data: pd.DataFrame):
    """Process NaN and mentions from comments in Tiktok."""
    print("Process mentions from comments in Tiktok!")
    
    mention_columns = [column for column in data.columns if column.startswith("mentions")]

    indexes_to_drop = []
    for index, row in data.iterrows():
        text = row["text"]

        if pd.isna(text):
            indexes_to_drop.append(index)
            continue

        mentions = [str(row[mention_column]) for mention_column in mention_columns if pd.notna(row[mention_column])]

        if mentions:
            total_len_mentions = sum(len(mention) for mention in mentions)
            if abs(len(text) - total_len_mentions) <= len(mentions):
                indexes_to_drop.append(index)
            else:
                text = text.replace("@", "").strip()
                data.at[index, "text"] = text
    
    return data.drop(index=indexes_to_drop).reset_index(drop=True)["text"]

def process_youtube_comments(data: pd.DataFrame):
    """Process NaN and mentions from comments in Youtube."""
    print("Process mentions from comments in Youtube!")
    
    mention_pattern = re.compile(r"@(\S+)")
    indexes_to_drop = []
    for index, row in data.iterrows():
        text = row["text"]
        
        if pd.isna(text):
            indexes_to_drop.append(index)
            continue
        
        text = mention_pattern.sub(r"\1", str(text))
        data.at[index, "text"] = text
    
    return data.drop(index=indexes_to_drop).reset_index(drop=True)["text"]

def process_facebook_comments(data: pd.DataFrame):
    """Process NaN comments in Facebook."""
    print("Process mentions from comments in Facebook!")
    
    indexes_to_drop = []
    for index, row in data.iterrows():
        text = row["text"]
    
        if pd.isna(text):
            indexes_to_drop.append(index)
    
    return data.drop(index=indexes_to_drop).reset_index(drop=True)["text"]

def process_reddit_comments(data: pd.DataFrame):
    """Process removed comments and comments that contain bot-generated messages, error emoji in Reddit."""
    print("Process error comments in Reddit!")
    
    data = data[(data["body"] != "[removed]") & (data["body"] != "") & (data["body"] != "[deleted]")]

    bot_pattern = r"\*I am a bot, and this action was performed automatically.*\*"
    data.loc[:, "body"] = data["body"].apply(lambda x: re.sub(bot_pattern, "", str(x)))

    image_pattern = r"!\[img\]\(emote\|[^\|]+\|[^\)]+\)"
    data.loc[:, "body"] = data["body"].apply(lambda x: re.sub(image_pattern, "", str(x)))

    return data["body"]

def add_id(data: List[Dict[str, Any]]):
    """Add unique IDs to each comment in the dataset."""
    print("Add unique IDs to each comment!")
    
    for i, item in tqdm(enumerate(data)):
        item["id"] = item["category"] + "_" + str(i)
    
    return data

def remove_duplicates(data: List[Dict[str, Any]]):
    """Remove duplicate comments."""
    print("Remove duplicate comments!")

    seen = set()
    new_data = []
    for item in tqdm(data):
        text = item["text"]
        if text not in seen:
            new_data.append(item)
            seen.add(text)

    return new_data

def remove_hashtags(data: List[Dict[str, Any]]):
    """Remove hashtags from comments."""
    print("Remove hashtags from comments!")
    
    hashtag_pattern = re.compile(r"#\S*")
    for item in tqdm(data):
        item["text"] = hashtag_pattern.sub("", item["text"])
    
    return data

def remove_urls(data: List[Dict[str, Any]]):
    """Remove URLs and image links from comments."""
    print("Remove URLs and image links from comments!")

    pattern = re.compile(r"https?://\S+|www\.\S+|!\[img\]\([^)]+\)|\[photo\]")
    for item in tqdm(data):
        item["text"] = pattern.sub("", item["text"])

    return data

def to_lowercase(data: List[Dict[str, Any]]):
    """Convert texts to lowercase."""
    print("Convert texts to lowercase!")
    
    for item in tqdm(data):
        item["text"] = item["text"].lower()
    
    return data

def clean_whitespace(data: List[Dict[str, Any]]):
    """Clean up extra whitespace in comments."""
    print("Clean up extra whitespace in comments!")
    
    for item in tqdm(data):
        item["text"] = re.sub(r"\s+", " ", item["text"]).strip()
    
    return data

def filter_non_vietnamese_texts(
    data: List[Dict[str, Any]],
    log_folder: str,
    batch_size: int = 100,
    flag: bool = False
):
    """Filter comments to keep only those in Vietnamese."""
    os.makedirs(log_folder, exist_ok=True)

    if flag == False:
        os.makedirs(os.path.join(log_folder, "detected_data"), exist_ok=True)
        new_data = []
        undetected_data = []
        for item in tqdm(data):
            try:
                if detect(item["text"]) == "vi":
                    new_data.append(item)
                else:
                    undetected_data.append(item)
            except LangDetectException:
                continue
        
        save_json(new_data, os.path.join(log_folder, "detected_data", "log_0.json"))

        if undetected_data:
            os.makedirs(os.path.join(log_folder, "undetected_data"), exist_ok=True)

        for i in range(0, len(undetected_data), batch_size):
            batch = undetected_data[i:i + batch_size]
            save_json(batch, os.path.join(log_folder, "undetected_data", f"batch_{i//batch_size + 1}.json"))
    
    if os.path.exists(os.path.join(log_folder, "undetected_data", "batch_1.json")):
        client = OpenAI(api_key=API_KEY)
        system_prompt = ."""
You are a Vietnamese language detection expert.

Your task is to determine whether a given text contains recognizable and meaningful Vietnamese content — including teencode, slang, or phonetic approximations.

Decision rules:
- ACCEPT the text as Vietnamese if:
    - It contains Vietnamese words, names, or slang, even if:
        - Tone marks are missing or incorrect
        - Teencode (teen language), misspellings, or phonetic substitutions are used (e.g., "bayf", "datw")
        - Words are written without spaces but still recognizable
        - It mixes with English or emojis, as long as over 50% of the message is understandable in Vietnamese
        - It includes distorted, playful, or creative Vietnamese that a native speaker would likely understand

REJECT the text as NOT Vietnamese if:
    - It is mostly (over 50%) written in another foreign language such as Chinese, Japanese, Korean, Thai, etc.
    - It contains only emojis, symbols, numbers, or nonsense
    - It has just a few Vietnamese words embedded in a sentence where the main message is not in Vietnamese

Output instructions:
- If the message is Vietnamese → return an <<empty output>>
- If the message is NOT Vietnamese → return None
- Do not include explanations, formatting, or punctuation

Examples:
Input: adu
Output: <<empty>>

Input: Chi Le , I want kiss you now....há há
Output: None

Input: ມືງໄປເຮັດຫຍັງຢູ່ຫັ້ນ
Output: None

Input: Please don't eat my family
Output: None

Input: chắc cái áo phải là sile xxxxxxxxxxxxl
Output: <<empty>>

Input: chuppy❌ con voi✅
Output: <<empty>>

Input: bel noi bel đi bayf datw chuppy
Output: <<empty>>

Input: bellphi
Output: <<empty>>

Input: chubby pro max😳
Output: <<empty>>

Input: Say get
Output: <<empty>>

Input: Gayset
Output: <<empty>>

Input: Wow 🔥🔥🌈🔥 mantp
Output: None

Input: M̷ặ̷c̷ m̷à̷y̷ g̷i̷ố̷n̷g̷ g̷á̷i̷ m̷ẹ̷ m̷à̷y̷ g̷h̷ê̷
Output: <<empty>>

Input: Parky
Output: <<empty>>

Input: aiosimi
Output: None

Input: kid
Output: None

Input: 在某些地方，说越南人不好就像是对中国南方人（特别是北京以南的）抱有偏见一样， thật là tệ mà!
Output: None

Input: Simp simp simpppppp
Output: None

Input: Khi nó đã một lần là con quỷ thì nó mãi mãi là con quỷ. \nMy leader said.
Output: <<empty>>
        """
        print("Filter comments to keep only those in Vietnamese!")
        
        for path in os.listdir(os.path.join(log_folder, "undetected_data")):
            if os.path.exists(os.path.join(log_folder, "detected_data", f"log_{path.split('_')[-1]}")):
                continue
            else:
                undetected_data_batch = json.load(open(os.path.join(log_folder, "undetected_data", path), "r", encoding="utf-8"))
                new_data = []
                removed_data = []
                for item in tqdm(undetected_data_batch):
                    response = client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item["text"]},
                        ],
                        stream=False
                    )
                    if response.choices[0].message.content == "":
                        new_data.append(item)
                    else:
                        removed_data.append(item)
                
                save_json(new_data, os.path.join(log_folder, "detected_data", f"log_{path.split('_')[-1]}"))
                save_json(removed_data, os.path.join(log_folder, "detected_data", f"log_-{path.split('_')[-1]}"))

    all_detected_data = []
    for file in os.listdir(os.path.join(log_folder, "detected_data")):
        if "-" in file:
            continue
        detected_data = load_json(os.path.join(log_folder, "detected_data", file))
        all_detected_data.extend(detected_data)
    return all_detected_data
