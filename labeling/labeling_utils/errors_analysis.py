import os
import logging
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
import matplotlib.pyplot as plt

from utils import load_json, save_json


def measure_agreement(
    result_folder: str,
    label_type: str,
    agreement_folder: str,
    prompt_round: int,
    sample: int = None
):
    if label_type == "toxicity":
        all_labels = ["TOXIC", "NON-TOXIC"]
    elif label_type == "toxic_type":
        all_labels = ["HATE", "OFFENSIVE"]
    elif label_type == "expression_type":
        all_labels = ["IMPLICIT", "EXPLICIT", "REPORTED"]
    else:
        raise ValueError("Invalid label_type. Supported label_types are: 'toxicity', 'toxic_type', 'expression_type'.")
    
    labelled_data_file_names = [file_path for file_path in os.listdir(result_folder)]
    annotators = []
    for file_name in labelled_data_file_names:
        if file_name.endswith(".json"):
            annotators.append(file_name.split("_")[-1].split(".")[0])

    all_labelled_data = []
    all_texts = {}
    for i, file_name in enumerate(labelled_data_file_names):
        labelled_data = load_json(os.path.join(result_folder, file_name))
        mapping = {item["id"]: item["toxicity_fixed"] for item in labelled_data}
        all_labelled_data.append(mapping)
        
        if i == 0:
            for item in labelled_data:
                all_texts[item["id"]] = item["text"]

    all_data_labels = sorted({label for labelled_data in all_labelled_data for label in labelled_data.values()})

    if not set(all_data_labels).issubset(set(all_labels)):
        error_labels = list(set(all_data_labels) - set(all_labels))
        
        error_ids = []
        for i, labelled_data in enumerate(all_labelled_data):
            for id, label in labelled_data.items():
                if label in error_labels:
                    error_ids.append((annotators[i], id))
        
        raise ValueError(f"Invalid label detected. The invalid label is {error_labels}, at position {error_ids}")

    all_ids = sorted(all_labelled_data[0].keys())
    num_annotators = len(all_labelled_data)
    label_to_index = {label: i for i, label in enumerate(all_labels)}

    if num_annotators > 2:
        matrix = np.zeros((len(all_ids), len(all_labels)), dtype=int)
        for row, id in enumerate(all_ids):
            for annotator in range(num_annotators):
                label = all_labelled_data[annotator][id]
                col = label_to_index[label]
                matrix[row, col] += 1
        kappa_score = fleiss_kappa(matrix)

        print("Fleiss' Kappa agreement:", kappa_score)
        if sample:
            logging.basicConfig(
                filename=os.path.join(agreement_folder, "agreement.log"),
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            logging.info(f"Fleiss' Kappa agreement with prompt round {prompt_round} in sample {sample}: {kappa_score}")
        else:
            logging.info(f"Fleiss' Kappa agreement with prompt round {prompt_round} in all data: {kappa_score}")

    else:
        result_data = list(all_labelled_data[0].values())
        llms_annotated_data = list(all_labelled_data[1].values())

        kappa_score = cohen_kappa_score(result_data, llms_annotated_data)

        print("Cohen' Kappa agreement:", kappa_score)
        if sample:
            logging.basicConfig(
                filename=os.path.join(agreement_folder, "agreement.log"),
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            logging.info(f"Cohen' Kappa agreement with prompt round {prompt_round} in sample {sample}: {kappa_score}")
        else:
            logging.info(f"Cohen' Kappa agreement with prompt round {prompt_round} in all data: {kappa_score}")

    different_labelled_data = []
    if kappa_score <= 0.9:
        for id in all_ids:
            labels = [labelled_data[id] for labelled_data in all_labelled_data]
            
            if len(set(labels)) > 1:
                entry = {"id": id, "text": all_texts.get(id)}
                for i, label in enumerate(labels):
                    entry[f"{annotators[i]}_label"] = label
                different_labelled_data.append(entry)
    
    different_labelled_folder = os.path.join(result_folder, "different_labelled")
    os.makedirs(different_labelled_folder, exist_ok=True)
    save_json(different_labelled_data, os.path.join(different_labelled_folder, "different_labelled_data.json"))

def plot_different_labels_bar(file_path: str):
    different_data = load_json(file_path)

    different_counts = defaultdict(int)
    for item in different_data:
        category = item["id"].split("_")[0]
        different_counts[category] += 1

    categories = list(different_counts.keys())
    counts = [different_counts[category] for category in categories]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel("Label")
    plt.ylabel("Number of different labelled data")
    plt.title("Number of different labelled data in each label")
    plt.show()

def plot_diffence_percentage(file_path: str, sample_size: int = 300):
    different_data = load_json(file_path)

    plt.figure(figsize=(8, 5))
    plt.pie(
        [sample_size - len(different_data), len(different_data)],
        labels=["Similarity", "Difference"],
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Percentage of Similar and Different Items") 
    plt.axis("equal")
    plt.show()

def plot_diffence_percentage_each_category(file_path: str, sample_size_per_label: int = 50):
    different_data = load_json(file_path)

    different_counts = defaultdict(int)
    for item in different_data:
        category = item["id"].split("_")[0]
        different_counts[category] += 1

    categories = list(different_counts.keys())
    num_category = len(category)

    num_col = 2
    num_row = (num_category + 1) // 2
    
    fig, axes = plt.subplots(num_row, num_col, figsize=(7, 9))
    axes = axes.flatten()

    for i, category in enumerate(categories):
        num_difference = different_counts.get(category)
        num_similarity = sample_size_per_label - num_difference

        axes[i].pie(
            [num_similarity, num_difference],
            labels=["Similarity", "Difference"],
            autopct="%1.1f%%",
            colors=["#8fd9a8", "#f28e8e"]
        )
        axes[i].set_title(f"Category '{category}'")
        axes[i].axis("equal")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
