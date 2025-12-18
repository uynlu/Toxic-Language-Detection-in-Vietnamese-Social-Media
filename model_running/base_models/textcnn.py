# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class TextCNN(nn.Module):
#     def __init__(
#         self,
#         vocab_size
#         embedding_dim,
#         num_classes,
#         embedding_matrix,
#         filter_sizes=[2,3,5],
#         num_filters=32,
#         dropout=0.5
#     ):
#         super().__init__()

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
#         self.embedding.weight.requires_grad = True  # fine-tune

#         self.convs = nn.ModuleList([
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=num_filters,
#                 kernel_size=(fs, embedding_dim)
#             )
#             for fs in filter_sizes
#         ])

#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

#     def forward(self, input):
#         # x: (batch, seq_len)
#         x = self.embedding(x)            # (batch, seq_len, emb_dim)
#         x = x.unsqueeze(1)               # (batch, 1, seq_len, emb_dim)

#         conv_outs = []
#         for conv in self.convs:
#             c = F.elu(conv(x)).squeeze(3)        # (batch, num_filters, seq_len-fs+1)
#             p = F.max_pool1d(c, c.size(2)).squeeze(2)
#             conv_outs.append(p)

#         out = torch.cat(conv_outs, dim=1)
#         out = self.dropout(out)
#         logits = self.fc(out)
#         return logits
