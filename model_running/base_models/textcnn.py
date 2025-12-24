import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import load_json


class TextCNN(nn.Module):
    def __init__(
        self,
        embedding_file_path: str,
        word2idx_path: str,
        num_labels: int,
        dropout_rate: float,
        filter_sizes: list[int] = [2, 3, 5],
        num_filters: int = 32,
        freeze_flag: bool = True,
        embedding_dim: int = 300,
    ):
        super(TextCNN, self).__init__()

        self.word2idx = load_json(word2idx_path)
        self.create_embedding_matrix(embedding_file_path, embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(self.embedding_matrix, dtype=torch.float32),
            freeze=freeze_flag
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(file_size, embedding_dim)
            )
            for file_size in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, inputs):
        embedding = self.embedding(inputs)  # (batch, seq_len, embedding_dim)
        embedding = embedding.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)

        conv_features = []
        for conv in self.convs:
            conv_feature = F.elu(conv(embedding))  # (batch, num_filters, seq_len - filter_size + 1, 1)
            pooled_feature = F.max_pool2d(
                conv_feature,
                kernel_size=(conv_feature.size(2), 1)
            )   # (batch, num_filters, 1, 1)
            conv_features.append(pooled_feature)

        conv_outputs = torch.cat(conv_features, dim=1)  # (batch, num_filters * len(filter_sizes), 1, 1)
        conv_outputs = conv_outputs.view(conv_outputs.size(0), -1)
        dropped_output = self.dropout(conv_outputs)

        output = self.fc(dropped_output)  # (batch, num_classes)

        return F.softmax(output, dim=1)
    
    def create_embedding_matrix(
        self,
        embedding_file_path: str,
        embedding_dim: int
    ):
        embeddings_index = {}
        with open(embedding_file_path, encoding='utf8') as file:
            for line in file:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        self.embedding_matrix = np.zeros((len(self.word2idx), embedding_dim))
        
        for word, idx in self.word2idx.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[idx] = embedding_vector
