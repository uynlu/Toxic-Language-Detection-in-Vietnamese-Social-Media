import os
import logging
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score

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


# def plot_ratio_of_similarity_to_difference(file_path: str, label: str):
#     data = load_json(file_path)

#     similarity_count = 0
#     difference_count = 0
    
#     for item in data:
#         if item.get(f"{label}_deepseek") == None:
#             similarity_count += 1
#         else:
#             difference_count += 1

#     plt.figure(figsize=(8, 6))
#     plt.pie(
#         [similarity_count, difference_count],
#         labels=["Similar", "Different"],
#         autopct="%1.1f%%",
#         colors=sns.color_palette("pastel"),
#         startangle=140
#     )
#     plt.title("Ratio of Similarity to Difference") 
#     plt.axis("equal")
#     plt.show()


# def plot_number_of_similar_comments_with_checker_each_annotator(file_path: str, label: str):
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     deepseek_count = 0
#     _count = 0
#     _count = 0
    
#     for item in data:
#         if item.get(f"{label}_deepseek") != None:
#             if item.get(f"{label}_deepseek") == item.get(f"{label}"):
#                 deepseek_count += 1
#             if item.get(f"{label}_") == item.get(f"{label}"):
#                 _count += 1
#             if item.get(f"{label}_") == item.get(f"{label}_"):
#                 _count += 1

#     plt.figure(figsize=(8, 6))
#     sns.barplot(
#         x=["Deepseek", "_", "_"],
#         y=[deepseek_count, _count, _count],
#         palette=sns.color_palette("pastel")
#     )
#     plt.title("Number of comments that are similar to the checker")
#     plt.ylabel("Number of Comments")
#     plt.show()

# def plot_number_of_different_comments_each_category(file_path: str, label: str):
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     categories = {}
#     for item in data:
#         if item.get(f"{label}_deepseek") != None:
#             try:
#                 categories[item["category"]] += 1
#             except KeyError:
#                 categories[item["category"]] = 1

#     df = pd.DataFrame(list(categories.items()), columns=["Category", "Count"])
#     df = df.sort_values(by="Count", ascending=False)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(
#         x="Category",
#         y="Count",
#         data=df,
#         palette=sns.color_palette("pastel")
#     )
#     plt.title("Number of Different Comments in Each Category")
#     plt.ylabel("Number of Comments")
#     plt.xticks(rotation=45)
#     plt.show()

# def print_different_comments(file_path: str):
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)

#     return pd.DataFrame(data)