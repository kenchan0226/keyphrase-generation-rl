import torch.nn as nn
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import Statistics
from utils.time_log import time_since
import time

def train_one_batch(batch, model, optimizer, opt):
    if opt.one2many:
        src, src_lens, src_mask, src_oov, oov_lists, src_str, trg_str, trg, trg_oov, trg_lens, trg_mask = batch
    else:
        src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists = batch
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
    trg_lens: a list containing the length of trg sequences for each batch, with len=batch
    trg_mask: a FloatTensor, [batch, trg_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    trg_oov: a LongTensor containing the word indices of target sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    :return:
    """

    max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    trg = trg.to(opt.device)
    trg_mask = trg_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    trg_oov = trg_oov.to(opt.device)

    model.train()

    optimizer.zero_grad()

    start_time = time.time()
    decoder_dist, h_t, attention_dist, coverage = model(src, src_lens, trg, src_oov, max_num_oov, src_mask)
    forward_time = time_since(start_time)

    start_time = time.time()
    if opt.copy_attention:  # Compute the loss using target with oov words
        loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                         opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)
    else:  # Compute the loss using target without oov words
        loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                    opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)
    loss_compute_time = time_since(start_time)

    total_trg_tokens = sum(trg_lens)

    if opt.loss_normalization == "tokens": # use number of target tokens to normalize the loss
        normalization = total_trg_tokens
    elif opt.loss_normalization == 'batches': # use batch_size to normalize the loss
        normalization = src.size(0)
    else:
        raise ValueError('The type of loss normalization is invalid.')

    assert normalization > 0, 'normalization should be a positive number'

    start_time = time.time()
    # back propagation on the normalized loss
    loss.div(normalization).backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        # grad_norm_after_clipping = (sum([p.grad.data.norm(2) ** 2 for p in model.parameters() if p.grad is not None])) ** (1.0 / 2)
        # logging.info('clip grad (%f -> %f)' % (grad_norm_before_clipping, grad_norm_after_clipping))

    optimizer.step()

    # construct a statistic object for the loss
    stat = Statistics(loss.item(), total_trg_tokens, n_batch=1, forward_time=forward_time, loss_compute_time=loss_compute_time, backward_time=backward_time)

    return stat, decoder_dist.detach()
