import json
import os
import re
from tqdm import tqdm
import random
from dotenv import load_dotenv
from openai import OpenAI

from utils import load_json, save_json


load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

MISTRAL_QWEN_API_KEY = os.getenv("MISTRAL_QWEN_API_KEY")
MISTRAL_QWEN_BASE_URL = os.getenv("MISTRAL_QWEN_BASE_URL")

GPT_API_KEY = os.getenv("GPT_API_KEY")


class DataAnnotatorPipeline:
    def __init__(
        self,
        label_type: str, # toxicity, toxic_type, expression_type
        annotating_system_prompt_path: str,
        checking_system_prompt_path: str,
        data_path: str,
        output_folder: str,
        prompt_round: int,
        result_path: str = None,
        optimization_flag: bool = False
    ):
        if optimization_flag == False and result_path is not None:
            raise ValueError("Cannot set result_path when optimization_flag is False.")

        if prompt_round == 1 and result_path is not None:
            raise ValueError("result_path is not allowed when prompt_round is 1.")
        
        if prompt_round != 1 and result_path is None:
            raise ValueError("result_path must be provided when prompt_round is not 1.")
        
        with open(annotating_system_prompt_path, "r", encoding="utf-8") as file:
            self.annotating_system_prompt = file.read()

        with open(checking_system_prompt_path, "r", encoding="utf-8") as file:
            self.checking_system_prompt = file.read()
            
        self.prompt_round = prompt_round
        os.makedirs(output_folder, exist_ok=True)
        self.batch_folder = os.path.join(output_folder, "batches")
        os.makedirs(os.path.join(output_folder, "batches"), exist_ok=True)
        self.result_folder = os.path.join(output_folder, f"round_{self.prompt_round}")
        os.makedirs(self.result_folder, exist_ok=True)

        with open(data_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)
        self.split_into_batches(100)
        self.batch_paths = [os.path.join(self.batch_folder, file) for file in os.listdir(self.batch_folder)]

        if label_type not in ["toxicity", "toxic_type", "expression_type"]:
            raise ValueError("Invalid label_type. Supported label_types are: 'toxicity', 'toxic_type', 'expression_type'.")
        self.label_type = label_type

        self.optimization_flag = optimization_flag
        self.result_path = result_path

        self.deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.mistral_qwen_client = OpenAI(api_key=MISTRAL_QWEN_API_KEY, base_url=MISTRAL_QWEN_BASE_URL)
        self.gpt_client = OpenAI(api_key=GPT_API_KEY)
        
    def main(self):
        if self.optimization_flag:
            print("Start prompt optimization process!")

            print("Start annotating!")
            self.annotate()

            print("Start checking!")
            self.check()
            
            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder}.json"), "r", encoding="utf-8"))

                labelled_data.extend(similar_data)
                labelled_data.extend(checked_data)
            
            fixed_labelled_data = self.create_fixed_labelled_data(labelled_data, 100)
            os.makedirs(os.path.join(self.result_folder, "result"), exist_ok=True)
            save_json(fixed_labelled_data, os.path.join(self.result_folder, "result", "labelled_data_llms.json"))
        else:
            print("Start annotation process!")
            print("Start annotating!")
            self.annotate()

            print("Start checking!")
            self.check()

            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder}.json"), "r", encoding="utf-8"))

                labelled_data.extend(similar_data)
                labelled_data.extend(checked_data)
            
            save_json(labelled_data, os.path.join(self.result_folder, "auto_labelled_data.json"))

    def annotate(self):
        for batch_path in self.batch_paths:
            print(f"Process {batch_path}!")
            saved_folder = os.path.join(self.result_folder, "annotated", f"annotated_{os.path.basename(batch_path).split('.')[0]}")
            if os.path.exists(os.path.join(saved_folder, "similar_data.json")) or os.path.exists(os.path.join(saved_folder, "different_data.json")):
                continue
            
            os.makedirs(saved_folder, exist_ok=True)
            
            with open(batch_path, "r", encoding="utf-8") as file:
                batch = json.load(file)

            similar_data = []
            different_data = []
            error_data = []
            for item in tqdm(batch):
                deepseek_response = self.deepseek_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": self.annotating_system_prompt},
                        {"role": "user", "content": item["text"]},
                    ],
                    stream=False
                )

                mistral_response = self.mistral_qwen_client.chat.completions.create(
                    model="mistralai/mistral-small-24b-instruct-2501",
                    messages=[
                        {"role": "system", "content": self.annotating_system_prompt},
                        {"role": "user", "content": item["text"]},
                    ],
                    stream=False
                )

                qwen_response = self.mistral_qwen_client.chat.completions.create(
                    model="qwen/qwen-plus",
                    messages=[
                        {"role": "system", "content": self.annotating_system_prompt},
                        {"role": "user", "content": item["text"]},
                    ],
                    stream=False
                )

                if self.optimization_flag:
                    try:
                        deepseek_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", deepseek_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                        mistral_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", mistral_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                        qwen_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", qwen_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                    except json.JSONDecodeError:
                        error_data.append(item)
                        continue
                
                    if deepseek_response_result["label"] == mistral_response_result["label"] == qwen_response_result["label"]:
                        item[f"{self.label_type}"] = deepseek_response_result["label"]

                        item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                        item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                        item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]

                        similar_data.append(item)
                    else:
                        item[f"{self.label_type}_deepseek"] = deepseek_response_result["label"]
                        item[f"{self.label_type}_mistral"] = mistral_response_result["label"]
                        item[f"{self.label_type}_qwen"] = qwen_response_result["label"]

                        item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                        item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                        item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]
                        
                        different_data.append(item)
                else:
                    deepseek_response_result = deepseek_response.choices[0].message.content
                    mistral_response_result = mistral_response.choices[0].message.content
                    qwen_response_result = qwen_response.choices[0].message.content
                    
                    if deepseek_response_result == mistral_response_result == qwen_response_result:
                        item[f"{self.label_type}"] = deepseek_response_result

                        similar_data.append(item)
                    else:
                        item[f"{self.label_type}_deepseek"] = deepseek_response_result
                        item[f"{self.label_type}_mistral"] = mistral_response_result
                        item[f"{self.label_type}_qwen"] = qwen_response_result

                        different_data.append(item)

            save_json(similar_data, os.path.join(saved_folder, "similar_data.json"))
            save_json(different_data, os.path.join(saved_folder, "different_data.json"))
            
            if error_data:
                save_json(error_data, os.path.join(saved_folder, "error_data.json"))
        
    def annotate_error_data(self, error_batch_folder: str):
        print("Process error data!")
        
        similar_data = load_json(os.path.join(error_batch_folder, "similar_data.json"))
        different_data = load_json(os.path.join(error_batch_folder, "different_data.json"))
        error_data = load_json(os.path.join(error_batch_folder, "error_data.json"))

        for item in error_data:
            print(item)
            deepseek_response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
                stream=False
            )

            mistral_response = self.mistral_qwen_client.chat.completions.create(
                model="mistralai/mistral-small-24b-instruct-2501",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
                stream=False
            )

            qwen_response = self.mistral_qwen_client.chat.completions.create(
                model="qwen/qwen-plus",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
                stream=False
            )

            if self.optimization_flag:
                print("Deepseek:", deepseek_response.choices[0].message.content)
                print("Mistral:", mistral_response.choices[0].message.content)
                print("Qwen:", qwen_response.choices[0].message.content)

                deepseek_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", deepseek_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                mistral_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", mistral_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                qwen_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", qwen_response.choices[0].message.content).strip(), re.DOTALL).group(0))
            
                if deepseek_response_result["label"] == mistral_response_result["label"] == qwen_response_result["label"]:
                    item[f"{self.label_type}"] = deepseek_response_result["label"]

                    item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                    item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                    item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]

                    similar_data.append(item)
                else:
                    item[f"{self.label_type}_deepseek"] = deepseek_response_result["label"]
                    item[f"{self.label_type}_mistral"] = mistral_response_result["label"]
                    item[f"{self.label_type}_qwen"] = qwen_response_result["label"]

                    item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                    item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                    item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]
                    
                    different_data.append(item)
            else:
                deepseek_response_result = deepseek_response.choices[0].message.content
                mistral_response_result = mistral_response.choices[0].message.content
                qwen_response_result = qwen_response.choices[0].message.content
                
                if deepseek_response_result == mistral_response_result == qwen_response_result:
                    item[f"{self.label_type}"] = deepseek_response_result

                    similar_data.append(item)
                else:
                    item[f"{self.label_type}_deepseek"] = deepseek_response_result
                    item[f"{self.label_type}_mistral"] = mistral_response_result
                    item[f"{self.label_type}_qwen"] = qwen_response_result

                    different_data.append(item)

        save_json(similar_data, os.path.join(error_batch_folder, "similar_data.json"))
        save_json(different_data, os.path.join(error_batch_folder, "different_data.json"))

    def check(self):
        saved_folder = os.path.join(self.result_folder, "checked")
        os.makedirs(saved_folder, exist_ok=True)

        for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
            print(f"Process {annotated_batch_folder}!")

            annotated_batch_folder_path = os.path.join(self.result_folder, "annotated", annotated_batch_folder)
            
            if os.path.exists(os.path.join(saved_folder, f"checked_{annotated_batch_folder}.json")):
                continue

            if os.path.exists(os.path.join(annotated_batch_folder_path, "different_data.json")) is False:
                continue
            
            different_data = load_json(os.path.join(annotated_batch_folder_path, "different_data.json"))

            checked_data = []

            for item in tqdm(different_data):
                used_keys = {
                    "text",
                    "category",
                    f"{self.label_type}_deepseek",
                    f"{self.label_type}_mistral",
                    f"{self.label_type}_qwen"
                }

                temp_item = {key: value for key, value in item.items() if key in used_keys}
                
                gpt_response = self.gpt_client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": self.checking_system_prompt},
                        {"role": "user", "content": json.dumps(temp_item, ensure_ascii=False)},
                    ],
                    stream=False
                )
                
                if self.optimization_flag:
                    gpt_response_result = json.loads(gpt_response.choices[0].message.content)

                    item[f"{self.label_type}"] = gpt_response_result["label"]
                    item[f"{self.label_type}_gpt_reason"] = gpt_response_result["reason"]
                else:
                    del item[f"{self.label_type}_deepseek"]
                    del item[f"{self.label_type}_mistral"]
                    del item[f"{self.label_type}_qwen"]

                    item[f"{self.label_type}"] = gpt_response_result

                checked_data.append(item)

            save_json(checked_data, os.path.join(saved_folder, f"checked_{annotated_batch_folder}.json"))
            
    def split_into_batches(self, batch_size: int):
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            save_json(batch, os.path.join(self.batch_folder, f"batch_{i // batch_size + 1}.json"))

    def create_fixed_labelled_data(self, labelled_data: list[dict], num_samples: int):
        fixed_labelled_data = []
        if self.optimization_flag:
            if self.result_path is None:
                for item in labelled_data:
                    temp_item = item.copy()
                    temp_item[f"{self.label_type}_fixed"] = temp_item[f"{self.label_type}"]
                    temp_item[f"{self.label_type}_fixed_reason"] = ""
                    fixed_labelled_data.append(temp_item)
            else:
                result_data = load_json(self.result_path)
                result_dict = {item["id"]: item for item in result_data}
        
                for item in labelled_data:
                    temp_item = item.copy()
                    item_id = item["id"]
                    temp_item["toxicity_fixed"] = result_dict[item_id]["toxicity_fixed"]
                    temp_item["toxicity_fixed_reason"] = result_dict[item_id]["toxicity_fixed_reason"]
                    fixed_labelled_data.append(temp_item)
        else:
            random_labelled_data = random.sample(labelled_data, num_samples)
            for item in random_labelled_data:
                temp_item = item.copy()
                temp_item[f"{self.label_type}_fixed"] = temp_item[f"{self.label_type}"]
                temp_item[f"{self.label_type}_fixed_reason"] = ""
                fixed_labelled_data.append(temp_item)

        return fixed_labelled_data
    