import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        cache_dir: str = None,
        freeze_model: bool = True,
        dropout_rate: float = 0.1
    ):
        super(PretrainedModel, self).__init__()
        
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, output_attentions=True)
        
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(self.model.config.hidden_size, num_labels)
    
    def forward(self, **input):
        """Forward pass through the model."""
        if self.model.name_or_path == "VietAI/vit5-base":
            model_output = self.model.get_encoder()(**input)
        else:
            model_output = self.model(**input)
            attentions = model_output.attentions if self.model.name_or_path != "VietAI/vit5-base" else None

        cls_hidden_state = model_output.last_hidden_state[:, 0, :]

        dropped_features = self.dropout(cls_hidden_state)
        logits = self.fc(dropped_features)

        return logits, attentions
    