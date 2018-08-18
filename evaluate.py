#from nltk.stem.porter import *
import torch
#from utils import Progbar
#from pykp.metric.bleu import bleu
from pykp.masked_loss import masked_cross_entropy
from utils.statistics import Statistics

def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src, src_lens, src_mask, trg, trg_lens, trg_mask, src_oov, trg_oov, oov_lists = batch

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch

            # move data to GPU if available
            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            trg = trg.to(opt.device)
            trg_mask = trg_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)
            trg_oov = trg_oov.to(opt.device)

            decoder_dist, h_t, attention_dist, coverage = model(src, src_lens, trg, src_oov, max_num_oov, src_mask)

            if opt.copy_attention:  # Compute the loss using target with oov words
                loss = masked_cross_entropy(decoder_dist, trg_oov, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)
            else:  # Compute the loss using target without oov words
                loss = masked_cross_entropy(decoder_dist, trg, trg_mask, trg_lens,
                                            opt.coverage_attn, coverage, attention_dist, opt.lambda_coverage)

            evaluation_loss_sum += loss.item()
            total_trg_tokens += sum(trg_lens)

    eval_loss_stat = Statistics(evaluation_loss_sum, total_trg_tokens)

    return eval_loss_stat
