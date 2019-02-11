import logging
import torch
import torch.nn as nn
import math
import logging
from pykp.masked_softmax import MaskedSoftmax


class RNNEncoder(nn.Module):
    """
    Base class for rnn encoder
    """
    def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
        raise NotImplementedError


class RNNEncoderBasic(RNNEncoder):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoderBasic, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True, dropout=dropout)

    def forward(self, src, src_lens, src_mask=None, title=None, title_lens=None, title_mask=None):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        # Debug
        #if math.isnan(self.rnn.weight_hh_l0[0,0].item()):
        #    logging.info('nan encoder parameter')
        src_embed = self.embedding(src) # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        memory_bank, encoder_final_state = self.rnn(packed_input_src)
        # ([batch, seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        memory_bank, _ = nn.utils.rnn.pad_packed_sequence(memory_bank, batch_first=True) # unpack (back to padded)

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1,:,:], encoder_final_state[-2,:,:]), 1) # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :] # [batch, hidden_size]

        return memory_bank.contiguous(), encoder_last_layer_final_state


class RNNEncoderTG(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, bidirectional, pad_token, dropout=0.0):
        super(RNNEncoderTG, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pad_token = pad_token
        #self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embed_size,
            self.pad_token
        )
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.merging_layer = nn.GRU(input_size=2*self.num_directions*hidden_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True)
        self.title_encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True)
        self.source_encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True)
        self.softmax = MaskedSoftmax(dim=2)
        self.match_fc = nn.Linear(self.num_directions * hidden_size, self.num_directions * hidden_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.res_ratio = 0.5

    def forward(self, src, src_lens, src_mask, title, title_lens, title_mask):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        :return:
        """
        # the src text pass through the source encoder to produce source context
        src_embed = self.embedding(src)  # [batch, src_len, embed_size]
        packed_src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True)
        src_context, _ = self.source_encoder(packed_src_embed)
        # ([batch, src_seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        src_context, _ = nn.utils.rnn.pad_packed_sequence(src_context, batch_first=True)  # unpack (back to padded)

        # add for residual connection
        res_src_context = src_context

        # the title text pass through the title encoder to produce title context
        title_embed = self.embedding(title)  # [batch, title_len, embed_size]
        # sort title according to length in a descending order
        title_lens_tensor = torch.LongTensor(title_lens).to(title.device)
        sorted_title_lens_tensor, title_idx_sorted = torch.sort(title_lens_tensor, dim=0, descending=True)
        _, title_idx_original = torch.sort(title_idx_sorted, dim=0)
        sorted_title_embed = title_embed.index_select(0, title_idx_sorted)
        packed_sorted_title_embed = nn.utils.rnn.pack_padded_sequence(sorted_title_embed, sorted_title_lens_tensor.tolist(), batch_first=True)
        title_context, _ = self.title_encoder(packed_sorted_title_embed)
        # ([batch, src_seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        title_context, _ = nn.utils.rnn.pad_packed_sequence(title_context, batch_first=True)  # unpack (back to padded)
        title_context = title_context.index_select(0, title_idx_original)

        attn_matched_seq = self.attn_matched_seq(src_context, title_context, title_mask)

        src_context = torch.cat([src_context, attn_matched_seq], dim=-1)
        src_context_dropouted = self.dropout(src_context)

        # final merge layer
        packed_src_context = nn.utils.rnn.pack_padded_sequence(src_context_dropouted, src_lens, batch_first=True)
        merged_src_context, encoder_final_state = self.merging_layer(packed_src_context)
        # ([batch, src_seq_len, num_directions*hidden_size], [num_layer * num_directions, batch, hidden_size])
        merged_src_context, _ = nn.utils.rnn.pad_packed_sequence(merged_src_context, batch_first=True)  # unpack (back to padded)

        # add residual connection
        final_src_context = self.res_ratio * res_src_context + (1 - self.res_ratio) * merged_src_context

        # only extract the final state in the last layer
        if self.bidirectional:
            encoder_last_layer_final_state = torch.cat((encoder_final_state[-1, :, :], encoder_final_state[-2, :, :]),
                                                       1)  # [batch, hidden_size*2]
        else:
            encoder_last_layer_final_state = encoder_final_state[-1, :, :]  # [batch, hidden_size]

        return final_src_context.contiguous(), encoder_last_layer_final_state

    def attn_matched_seq(self, src_context, title_context, title_mask):
        """
        :param src_context: [batch, src_seq_len, num_directions*hidden_size]
        :param title_context: [batch, title_seq_len, num_directions*hidden_size]
        :return:
        """
        src_seq_len = src_context.size(1)
        # compute score
        matched_title_context = self.match_fc(title_context)  # [batch, title_seq_len, num_directions * hidden_size]
        scores = src_context.bmm(matched_title_context.transpose(2, 1))  # [batch, src_seq_len, title_seq_len]
        expanded_title_mask = title_mask.unsqueeze(1).expand(-1, src_seq_len, -1)  # [batch, src_seq_len, title_seq_len]

        # normalize the score
        attn_dist = self.softmax(scores, mask=expanded_title_mask)  # [batch, src_seq_len, title_seq_len]

        # compute weighted sum
        matched_src_context = attn_dist.bmm(title_context)  # [batch, src_seq_len, num_directions * hidden_size]

        return matched_src_context

