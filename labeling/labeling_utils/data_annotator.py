import json
import os
import re
from tqdm import tqdm
import random
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import load_json, save_json


load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_QWEN_API_KEY = os.getenv("MISTRAL_QWEN_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")

MAX_WORKERS = 64


class DataAnnotatorPipeline:
    def __init__(
        self,
        label_type: str, # toxicity, toxic_type, expression_type
        annotating_system_prompt_path: str,
        checking_system_prompt_path: str,
        # data_path: str,
        output_folder: str,
        prompt_round: int,
        optimization_flag: bool = False
    ):
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

        if not glob.glob(os.path.join(self.batch_folder, "*.json")):
            self.data = load_json(data_path)
            self.split_into_batches(100)
        
        self.batch_paths = [os.path.join(self.batch_folder, file_name) for file_name in os.listdir(self.batch_folder)]

        if label_type not in ["toxicity", "toxic_type", "expression_type"]:
            raise ValueError("Invalid label_type. Supported label_types are: 'toxicity', 'toxic_type', 'expression_type'.")
        self.label_type = label_type

        self.optimization_flag = optimization_flag

        self.error_flag = False

        self.deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        self.mistral_qwen_client = OpenAI(api_key=MISTRAL_QWEN_API_KEY, base_url="https://openrouter.ai/api/v1")
        self.gpt_client = OpenAI(api_key=GPT_API_KEY)
        
    def main(self):
        if self.optimization_flag:
            print("Start prompt optimization process!")

            print("Start annotating!")
            self.annotate()

            if self.error_flag:
                return
            
            print("Start checking!")
            self.check()
            
            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder[10:]}.json"), "r", encoding="utf-8"))

                labelled_data.extend(similar_data)
                labelled_data.extend(checked_data)
            
            fixed_labelled_data = self.create_fixed_labelled_data(labelled_data, 100)
            os.makedirs(os.path.join(self.result_folder, "result"), exist_ok=True)
            save_json(fixed_labelled_data, os.path.join(self.result_folder, "result", "labelled_data_llms.json"))
        else:
            print("Start annotation process!")
            print("Start annotating!")
            self.annotate()

            if self.error_flag:
                return
            
            print("Start checking!")
            self.check()

            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder[10:]}.json"), "r", encoding="utf-8"))

                labelled_data.extend(similar_data)
                labelled_data.extend(checked_data)
            
            save_json(labelled_data, os.path.join(self.result_folder, "auto_labelled_data.json"))

    def main_parallel(self):
        if self.optimization_flag:
            print("Start prompt optimization process!")

            print("Start annotating!")
            self.annotate_parallelly()

            if self.error_flag:
                return
            
            print("Start checking!")
            self.check_parallelly()
            
            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder[10:]}.json"), "r", encoding="utf-8"))

                labelled_data.extend(similar_data)
                labelled_data.extend(checked_data)
            
            fixed_labelled_data = self.create_fixed_labelled_data(labelled_data, 100)
            os.makedirs(os.path.join(self.result_folder, "result"), exist_ok=True)
            save_json(fixed_labelled_data, os.path.join(self.result_folder, "result", "labelled_data_llms.json"))
        else:
            print("Start annotation process!")
            print("Start annotating!")
            self.annotate_parallelly()

            return
            
            print("Start checking!")
            self.check_parallelly()

            labelled_data = []
            for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
                similar_data = json.load(open(os.path.join(self.result_folder, "annotated", annotated_batch_folder, "similar_data.json"), "r", encoding="utf-8"))
                checked_data = json.load(open(os.path.join(self.result_folder, "checked", f"checked_{annotated_batch_folder[10:]}.json"), "r", encoding="utf-8"))

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
            
            batch = load_json(batch_path)

            similar_data = []
            different_data = []
            error_data = []
            for item in tqdm(batch):
                try:
                    deepseek_response = self.deepseek_client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": self.annotating_system_prompt},
                            {"role": "user", "content": item["text"]},
                        ],
                        stream=False
                    )

                    mistral_response = self.mistral_qwen_client.chat.completions.create(
                        model="mistralai/mistral-large-2512",
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
                except:
                    self.error_flag = True
                    error_data.append(item)
                    continue
                
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

                try:
                    deepseek_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", deepseek_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                    mistral_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", mistral_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                    qwen_response_result = json.loads(re.search(r"\{.*\}", re.sub(r"```json|```", "", qwen_response.choices[0].message.content).strip(), re.DOTALL).group(0))
                except:
                    raw = re.sub(r"```json|```", "", mistral_response.choices[0].message.content).strip()
                    raw = re.sub(r'("reason"\s*:\s*")([\s\S]*?)"', lambda m: m.group(1) + m.group(2).replace("\n", "\\n") + '"', raw)
                    qwen_response_result = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))

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

    def annotate_parallelly(self):
        for batch_path in self.batch_paths:
            print(f"Process {batch_path}!")
            saved_folder = os.path.join(self.result_folder, "annotated", f"annotated_{os.path.basename(batch_path).split('.')[0]}")
            if os.path.exists(os.path.join(saved_folder, "similar_data.json")) or os.path.exists(os.path.join(saved_folder, "different_data.json")):
                continue
            
            os.makedirs(saved_folder, exist_ok=True)
            
            batch = load_json(batch_path)

            similar_data = []
            different_data = []
            error_data = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {executor.submit(self.annotate_one_item, item): item for item in batch}
                
                for future in tqdm(as_completed(future_to_item), total=len(batch)):
                    annotated_item, status = future.result()
                    if status == "similar":
                        similar_data.append(annotated_item)
                    elif status == "different":
                        different_data.append(annotated_item)
                    else:
                        error_data.append(annotated_item)
                        self.error_flag = True
                
            save_json(similar_data, os.path.join(saved_folder, "similar_data.json"))
            save_json(different_data, os.path.join(saved_folder, "different_data.json"))
            
            if error_data:
                save_json(error_data, os.path.join(saved_folder, "error_data.json"))
    
    # def annotate_error_data_parallelly(self, error_batch_folder: str):
    #     print("Process error data!")
        
    #     similar_data = load_json(os.path.join(error_batch_folder, "similar_data.json"))
    #     different_data = load_json(os.path.join(error_batch_folder, "different_data.json"))
    #     error_data = load_json(os.path.join(error_batch_folder, "error_data.json"))

    #     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #         future_to_item = {executor.submit(self.annotate_one_item, item): item for item in error_data}

    #         for future in tqdm(as_completed(future_to_item), total=len(error_data)):
    #             annotated_item, status, deepseek_response, mistral_response, qwen_response = future.result()
    #             if status == "similar":
    #                 similar_data.append(annotated_item)
    #             elif status == "different":
    #                 different_data.append(annotated_item)
    #             else:
    #                 print(deepseek_response)
    #                 print(mistral_response)
    #                 print(qwen_response)
    #                 raise ValueError(f"Error in annotating item again! Item: {annotated_item}")
        
    #     save_json(similar_data, os.path.join(error_batch_folder, "similar_data.json"))
    #     save_json(different_data, os.path.join(error_batch_folder, "different_data.json"))

    def annotate_one_item(self, item: dict):
        try:
            deepseek_response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
            )
            mistral_response = self.mistral_qwen_client.chat.completions.create(
                model="mistralai/mistral-large-2512",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
            )
            qwen_response = self.mistral_qwen_client.chat.completions.create(
                model="qwen/qwen-plus",
                messages=[
                    {"role": "system", "content": self.annotating_system_prompt},
                    {"role": "user", "content": item["text"]},
                ],
            )

            if self.optimization_flag:
                deepseek_response_result = json.loads(re.search(r"\{.*\}", deepseek_response.choices[0].message.content, re.DOTALL).group(0))
                mistral_response_result = json.loads(re.search(r"\{.*\}", mistral_response.choices[0].message.content, re.DOTALL).group(0))
                qwen_response_result = json.loads(re.search(r"\{.*\}", qwen_response.choices[0].message.content, re.DOTALL).group(0))

                if deepseek_response_result["label"] == mistral_response_result["label"] == qwen_response_result["label"]:
                    item[f"{self.label_type}"] = deepseek_response_result["label"]
                    
                    item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                    item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                    item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]
                    
                    return item, "similar"
                else:
                    item[f"{self.label_type}_deepseek"] = deepseek_response_result["label"]
                    item[f"{self.label_type}_mistral"] = mistral_response_result["label"]
                    item[f"{self.label_type}_qwen"] = qwen_response_result["label"]
                    
                    item[f"{self.label_type}_deepseek_reason"] = deepseek_response_result["reason"]
                    item[f"{self.label_type}_mistral_reason"] = mistral_response_result["reason"]
                    item[f"{self.label_type}_qwen_reason"] = qwen_response_result["reason"]

                    return item, "different"
            else:
                deepseek_response_result = deepseek_response.choices[0].message.content
                mistral_response_result  = mistral_response.choices[0].message.content
                qwen_response_result = qwen_response.choices[0].message.content

                if deepseek_response_result == mistral_response_result == qwen_response_result:
                    item[f"{self.label_type}"] = deepseek_response_result
                    return item, "similar"
                else:
                    item[f"{self.label_type}_deepseek"] = deepseek_response_result
                    item[f"{self.label_type}_mistral"] = mistral_response_result
                    item[f"{self.label_type}_qwen"] = qwen_response_result
                    
                    return item, "different"

        except:
            return item, "error"

    def check(self):
        saved_folder = os.path.join(self.result_folder, "checked")
        os.makedirs(saved_folder, exist_ok=True)

        for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
            annotated_batch_folder_path = os.path.join(self.result_folder, "annotated", annotated_batch_folder)
            print(f"Process {annotated_batch_folder_path}!")

            if os.path.exists(os.path.join(saved_folder, f"checked_{annotated_batch_folder[10:]}.json")):
                continue

            if not os.path.exists(os.path.join(annotated_batch_folder_path, "different_data.json")):
                continue
            
            different_data = load_json(os.path.join(annotated_batch_folder_path, "different_data.json"))

            checked_data = []
            for item in tqdm(different_data):
                used_keys = {
                    "text",
                    # "category",
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

            save_json(checked_data, os.path.join(saved_folder, f"checked_{annotated_batch_folder[10:]}.json"))

    def check_parallelly(self):
        saved_folder = os.path.join(self.result_folder, "checked")
        os.makedirs(saved_folder, exist_ok=True)

        for annotated_batch_folder in os.listdir(os.path.join(self.result_folder, "annotated")):
            annotated_batch_folder_path = os.path.join(self.result_folder, "annotated", annotated_batch_folder)
            print(f"Process {annotated_batch_folder_path}!")

            if os.path.exists(os.path.join(saved_folder, f"checked_{annotated_batch_folder[10:]}.json")):
                continue

            if not os.path.exists(os.path.join(annotated_batch_folder_path, "different_data.json")):
                continue
            
            different_data = load_json(os.path.join(annotated_batch_folder_path, "different_data.json"))

            checked_data = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_item = {executor.submit(self.check_one_item, item): item for item in different_data}
                
                for future in tqdm(as_completed(future_to_item), total=len(different_data)):
                    checked_item = future.result()
                    checked_data.append(checked_item)
                
            save_json(checked_data, os.path.join(saved_folder, f"checked_{annotated_batch_folder[10:]}.json"))

    def check_one_item(self, item: dict):
        used_keys = {
            "text",
            # "category",
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

        return item
        
    def split_into_batches(self, batch_size: int):
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            save_json(batch, os.path.join(self.batch_folder, f"batch_{i // batch_size + 1}.json"))

    def create_fixed_labelled_data(self, labelled_data: list[dict], num_samples: int):
        fixed_labelled_data = []
        if self.optimization_flag:
            for item in labelled_data:
                temp_item = item.copy()
                temp_item[f"{self.label_type}_fixed"] = temp_item[f"{self.label_type}"]
                temp_item[f"{self.label_type}_fixed_reason"] = ""
                fixed_labelled_data.append(temp_item)
        else:
            random_labelled_data = random.sample(labelled_data, num_samples)
            for item in random_labelled_data:
                temp_item = item.copy()
                temp_item[f"{self.label_type}_fixed"] = temp_item[f"{self.label_type}"]
                temp_item[f"{self.label_type}_fixed_reason"] = ""
                fixed_labelled_data.append(temp_item)

        return fixed_labelled_data
    
    # def annotate_temp(self):
    #     saved_folder = os.path.join(self.result_folder, "annotated")
    #     os.makedirs(saved_folder, exist_ok=True)
        
    #     data = []
    #     for batch_path in self.batch_paths:
    #         print(f"Process {batch_path}!")
            
    #         if os.path.exists(os.path.join(saved_folder, f"{os.path.basename(batch_path).split('.')[0]}.json")):
    #             continue
            
    #         batch = load_json(batch_path)
            
    #         data = []
    #         error_data = []
    #         for item in tqdm(batch):
    #             try:
    #                 gpt_response = self.gpt_client.chat.completions.create(
    #                     model="gpt-5",
    #                     messages=[
    #                         {"role": "system", "content": self.annotating_system_prompt},
    #                         {"role": "user", "content": item["text"]},
    #                     ],
    #                     stream=False
    #                 )
                    
    #                 item[f"{self.label_type}"] = gpt_response.choices[0].message.content
    #                 data.append(item)

    #             except:
    #                 self.error_flag = True
    #                 error_data.append(item)
    #                 continue
                
    #         save_json(data, os.path.join(saved_folder, f"{os.path.basename(batch_path).split('.')[0]}.json"))

    #         if error_data:
    #             save_json(error_data, os.path.join(saved_folder, f"error_{os.path.basename(batch_path).split('.')[0]}.json"))
