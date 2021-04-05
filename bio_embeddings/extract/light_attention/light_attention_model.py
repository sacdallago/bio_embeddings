import torch.nn as nn
import torch


class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=10, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: torch.Tensor [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified

            :return: [batch_size,output_dim] tensor with logits
        """

        intermediate_state = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        intermediate_state = self.dropout(intermediate_state)  # [batch_size, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # take the weighted sum according to the attention scores
        attention_pooled = torch.sum(intermediate_state * self.softmax(attention),
                                     dim=-1)  # [batchsize, embeddings_dim]

        # take the maximum over the length dimension
        max_pooled, _ = torch.max(intermediate_state, dim=-1)  # [batchsize, embeddings_dim]

        intermediate_state = torch.cat([attention_pooled, max_pooled], dim=-1)  # [batchsize, 2*embeddings_dim]
        intermediate_state = self.linear(intermediate_state)  # [batchsize, 32]
        return self.output(intermediate_state)  # [batchsize, output_dim]
