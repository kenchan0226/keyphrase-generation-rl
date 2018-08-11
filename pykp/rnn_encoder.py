import logging
import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout=0.0):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src_embed, input_src_len):
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, input_src_len, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True) # unpack (back to padded)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state
