import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.amp import GradScaler
from tqdm import tqdm
import os
import shutil
import time

from dataset.custom_dataset import CustomDataset
from model_running.base_models import PretrainedModel
from model_running.base_models import TextCNN
from model_running.model_utils.evaluation_errors_utils import error
from utils import save_json


class ModelExecutor:
    def __init__(
        self,
        batch_size: int,
        model_name: str,
        num_labels: int,
        checkpoint_path: str,
        cache_dir: int = None,
        pretrained_flag: bool = True,
        freeze_flag: bool = True,
        dropout_rate: float = 0.1,
        num_epochs: int = 100,
        learning_rate: int = 2e-5,
        use_amp: bool = False,
        device: str = "cuda",
        embedding_file_path: str = None,
        word2idx_path: str = None,
    ):
        super(ModelExecutor, self).__init__()

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.pretrained_flag = pretrained_flag

        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        if pretrained_flag:
            self.model = PretrainedModel(
                model_name=model_name,
                num_labels=num_labels,
                cache_dir=cache_dir,
                freeze_model=freeze_flag,
                dropout_rate=dropout_rate
            ).to(self.device)
        else:
            if model_name == "TextCNN":
                self.model = TextCNN(
                    embedding_file_path=embedding_file_path,
                    word2idx_path=word2idx_path,
                    num_labels=num_labels,
                    dropout_rate=dropout_rate,
                    freeze_flag=freeze_flag
                ).to(self.device)

        self.num_labels = num_labels
        self.criterion = CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.grad_scaler = GradScaler(enabled=use_amp)
        self.epoch = 1
        self.patience = 0
        self.num_epochs = num_epochs

    def create_loader(
        self,
        train_dataset: CustomDataset,
        validation_dataset: CustomDataset,
        test_dataset: CustomDataset
    ):
        self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=1)
        self.test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    def train(self):
        """Train model."""
        self.model.train()

        running_loss = 0.0
        with tqdm(desc="Epoch %d - Training" % self.epoch, unit="it", total=len(self.train_loader)) as pbar:
            for i, batch in enumerate(self.train_loader, start=1):
                if self.pretrained_flag:
                    inputs = {
                        "input_ids": batch["input_ids"].to(self.device),
                        "attention_mask": batch["attention_mask"].to(self.device)
                    }
                else:
                    inputs = batch["input_ids"].to(self.device)

                labels = batch["label"]
                labels = labels.to(self.device)
                
                logits = self.model(inputs)

                loss = self.criterion(logits, labels)
                
                this_loss = loss.item()
                running_loss += this_loss

                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                pbar.set_postfix(loss=f"{running_loss / i}")
                pbar.update()
        
        return running_loss / len(self.train_loader)
    
    def evaluate(
        self,
        type: str
    ):
        """Evaluate model."""
        if type == "validation":
            loader = self.validation_loader
        else:
            loader = self.test_loader
            test_results = []

        self.model.eval()

        predictions = []
        labels = []

        running_loss = 0.0
        with tqdm(desc="Epoch %d - Evaluation" % self.epoch, unit="it", total=len(loader)) as pbar:
            for i, batch in enumerate(loader, start=1):
                with torch.no_grad():
                    if self.pretrained_flag:
                        inputs = {
                            "input_ids": batch["input_ids"].to(self.device),
                            "attention_mask": batch["attention_mask"].to(self.device)
                        }
                    else:
                        inputs = batch["input_ids"].to(self.device)

                    label = batch["label"]

                    label = label.to(self.device)
                    
                    logits = self.model(inputs)
                    loss = self.criterion(logits, label)

                    this_loss = loss.item()
                    running_loss += this_loss

                prediction = torch.argmax(logits, dim=-1)

                predictions.append(prediction)
                labels.append(label)
                
                if type == "test":
                    entry = dict()
                    entry["text"] = batch["text"][0]
                    entry["prediction"] = prediction.tolist()[0]
                    entry["label"] = label.tolist()[0]
                    test_results.append(entry)

                pbar.set_postfix(loss=f"{running_loss / i}")
                pbar.update()
        
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        
        acc, f1, precision, recall = error(labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(), self.num_labels)
        print(f"Evaluation scores: Accuracy - {acc}, F1 score - {f1}, Precision - {precision}, Recall - {recall}")
        if type == "validation":
            return (
                predictions,
                labels,
                running_loss / len(loader),
                acc,
                f1,
                precision,
                recall
            )
        else:
            return (
                test_results,
                predictions,
                labels,
                running_loss / len(loader),
                acc,
                f1,
                precision,
                recall
            )

    def save_checkpoint(
        self,
        train_loss: float,
        validation_loss: float,
        validation_accuracy: float,
        validation_f1_score: float,
        validation_precision: float,
        validation_recall: float,
        training_time: float
    ):
        """Save checkpoint."""
        dict_for_saving = {
            "epoch": self.epoch,
            "patience": self.patience,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_f1_score": validation_f1_score,
            "validation_precision": validation_precision,
            "validation_recall": validation_recall,
            "training_time": training_time
        }

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def load_checkpoint(self, file_path: str):
        """Load checkpoint."""
        if not os.path.exists(file_path):
            return None
        print("Loading checkpoint from ", file_path)
        checkpoint = torch.load(file_path)
        return checkpoint

    def run(
        self,
        train_dataset: CustomDataset,
        validation_dataset: CustomDataset,
        test_dataset: CustomDataset,
        patience_threshold: int = 10
    ):
        """Run executor."""
        self.create_loader(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset
        )

        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            if os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
                best_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
                best_f1_score = best_checkpoint["validation_f1_score"]
            else:
                best_f1_score = -1
            
            last_checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            self.epoch = last_checkpoint["epoch"]
            self.model.load_state_dict(last_checkpoint["model_state_dict"], strict=False)
            self.optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
            self.patience = last_checkpoint["patience"]
            training_time = last_checkpoint["training_time"]
            
            print(f"Resuming from epoch {self.epoch}")
        else:
            training_time = 0
            best_f1_score = -1
        
        print("Start training!")
        while True:
            start_time = time.time()
            train_loss = self.train()
            end_time = time.time()
            
            best = False

            training_time += (end_time - start_time)

            (
                _,
                _,
                validation_loss,
                validation_accuracy,
                validation_f1_score,
                validation_precision,
                validation_recall
            ) = self.evaluate(type="validation")
            
            self.save_checkpoint(
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                validation_f1_score=validation_f1_score,
                validation_precision=validation_precision,
                validation_recall=validation_recall,
                training_time=training_time
            )

            if validation_f1_score > best_f1_score:
                best = True
                best_f1_score = validation_f1_score
                self.patience = 0
            else:
                self.patience += 1
            
            if best:
                shutil.copyfile(
                    os.path.join(self.checkpoint_path, "last_model.pth"), 
                    os.path.join(self.checkpoint_path, "best_model.pth")
                )

            if self.epoch == self.num_epochs or self.patience == patience_threshold:
                break

            self.epoch += 1
        print("Finish training!")

        print("Start testing!")
        checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_time = time.time()
        test_results, predictions, targets, _, _, _, _, _ = self.evaluate(type="test")
        end_time = time.time()

        elapsed = end_time - start_time

        accuracy, f1_score, precision, recall = error(predictions.detach().cpu().numpy(), targets.detach().cpu().numpy(), self.num_labels)

        save_json(
            {
                "time": elapsed,
                "predictions": predictions.tolist(),
                "targets": targets.tolist(),
                "results": test_results,
                "accuracy": accuracy,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall
            },
            os.path.join(self.checkpoint_path, "test_results.json")
        )

        print(f"Thời gian tính toán (test dataset): {elapsed:.2f} giây ({elapsed / 60:.2f} phút)")

    # def get_predictions(self, input_ids):
    #     if not os.path.isfile(os.path.join(self.checkpoint_path, "best_model.pth")):
    #         print("Prediction require the model must be trained. There is no weights to load for model prediction!")
    #         raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path!")

    #     checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))
    #     self.model.load_state_dict(checkpoint["model_state_dict"])

    #     self.model.eval()
    #     with torch.no_grad():
    #         input_ids = input_ids.to(self.device)
    #         logits = self.model(input_ids)

    #         prediction = torch.argmax(logits, dim=-1)

    #     print(prediction)
        
    #     save_json(
    #         {"prediction": prediction.detach().cpu().numpy().tolist()},
    #         os.path.join(self.checkpoint_path, "predictions.json"),
    #     )
