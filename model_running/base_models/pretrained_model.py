import torch
import torch.nn as nn
from transformers import AutoModel, MT5EncoderModel, AutoConfig


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
        if model_name == "google/mt5-base" or model_name == "google/mt5-large":
            self.model = MT5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir, output_attentions=True)
        elif model_name == "uitnlp/CafeBERT":
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            config.num_hidden_layers = 12
            config.hidden_dropout_prob = 0.3          # default: 0.1
            config.attention_probs_dropout_prob = 0.3 # default: 0.1
            config.classifier_dropout = 0.3 
            config.output_attentions = True
            self.model = AutoModel.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        else:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, output_attentions=True)

        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(self.model.config.hidden_size, num_labels)
    
    def forward(self, **input):
        """Forward pass through the model."""
        model_output = self.model(**input)
        
        attentions = model_output.attentions if (self.model.name_or_path != "vinai/bartpho-syllable" and self.model.name_or_path != "vinai/bartpho-word") else model_output.encoder_attentions

        attn_cls = attentions[-1].mean(1)[0, 0, :]

        cls_hidden_state = model_output.last_hidden_state[:, 0, :]

        dropped_features = self.dropout(cls_hidden_state)
        logits = self.fc(dropped_features)

        return logits, attn_cls
