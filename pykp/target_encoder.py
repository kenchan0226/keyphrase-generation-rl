import torch
import torch.nn as nn
from pykp.attention import Attention
from pykp.masked_softmax import MaskedSoftmax

class TargetEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, pad_idx):
        super(TargetEncoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.pad_token = pad_idx
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=1,
                          bidirectional=False)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )

    def forward(self, y, h):
        """
        :param y: [batch_size]
        :param h: [1, batch_size, target_encoder_size]
        :return:
        """
        y_emb = self.embedding(y).unsqueeze(0)  # [1, batch_size, embed_size]
        _, h_next = self.rnn(y_emb, h)  # [1, batch_size, target_encoder_size]
        return h_next

