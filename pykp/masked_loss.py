import torch


def masked_cross_entropy(class_dist, target, trg_mask, divided_by_seq_len=False, trg_lens=None,
                         coverage_attn=False, coverage=None, attn_dist=None, lambda_coverage=0):
    """
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :param divided_by_seq_len: boolean, whether to divide the loss by the max target sequence length
    :param trg_lens: a list with len of batch_size
    :param coverage_attn: boolean, whether to include coverage loss
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param lambda_coverage: scalar, coefficient for coverage loss
    :return:
    """
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.view(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    log_dist_flat = torch.log(class_dist_flat)
    target_flat = target.view(-1, 1)  # [batch*trg_seq_len, 1]
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat) # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]
    if coverage_attn:
        coverage_losses = compute_coverage_losses(coverage, attn_dist)
        losses = losses + lambda_coverage * coverage_losses
    if trg_mask is not None:
        losses = losses * trg_mask
    if divided_by_seq_len:
        trg_lens_tensor = torch.FloatTensor(trg_lens).to(target.device).requires_grad_()
        loss = losses.sum(dim=1)   # [batch_size]
        loss = loss / trg_lens_tensor
    else:
        loss = losses.sum(dim=1) # [batch_size]
    return loss.sum()

def compute_coverage_losses(coverage, attn_dist):
    """
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :return: coverage_losses: [batch, trg_seq_len]
    """
    batch_size = coverage.size(0)
    trg_seq_len = coverage.size(1)
    src_seq_len = attn_dist.size(2)
    coverage_flat = coverage.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    attn_dist_flat = attn_dist.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    coverage_losses_flat = torch.sum(torch.min(attn_dist_flat, coverage_flat), dim=1)  # [batch_size * trg_seq_len]
    coverage_losses = coverage_losses_flat.view(batch_size, trg_seq_len)  # [batch, trg_seq_len]
    return coverage_losses


def masked_coverage_loss(coverage, attn_dist, trg_mask):
    """
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :return:
    """
    src_seq_len = attn_dist.size(2)
    coverage_flat = coverage.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    attn_dist_flat = attn_dist.view(-1, src_seq_len)  # [batch_size * trg_seq_len, src_seq_len]
    coverage_losses_flat = torch.sum(torch.min(attn_dist_flat, coverage_flat), 1)  # [batch_size * trg_seq_len]
    coverage_losses = coverage_losses_flat.view(*trg_mask.size())  # [batch, trg_seq_len]
    if trg_mask is not None:
        coverage_losses = coverage_losses * trg_mask
    return coverage_losses.sum()


"""
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :param divided_by_seq_len: boolean, whether to divide the loss by the max target sequence length
    :param trg_lens: a list with len of batch_size
    :param coverage_attn: boolean, whether to include coverage loss
    :param coverage: [batch_size, trg_seq_len, src_seq_len]
    :param attn_dist: [batch_size, trg_seq_len, src_seq_len]
    :param lambda_coverage: scalar, coefficient for coverage loss
    :return:
    """
if __name__ == '__main__':
    import torch.nn.functional as F
    import numpy as np
    torch.manual_seed(1234)
    np.random.seed(1234)

    num_classes = 5000
    batch_size = 5
    trg_seq_len = 6
    src_seq_len = 30
    class_dist = torch.randint(0, 5, (batch_size, trg_seq_len, num_classes))
    class_dist = F.softmax(class_dist, dim=-1)

    target = np.random.randint(2, 300, (batch_size, trg_seq_len))
    target[batch_size-1, trg_seq_len-1] = 0
    target[batch_size - 1, trg_seq_len - 2] = 0
    target[batch_size - 2, trg_seq_len - 1] = 0
    target = torch.LongTensor(target)

    trg_mask = np.ones((batch_size, trg_seq_len))
    target[batch_size - 1, trg_seq_len - 1] = 0
    target[batch_size - 1, trg_seq_len - 2] = 0
    target[batch_size - 2, trg_seq_len - 1] = 0
    trg_mask = torch.FloatTensor(trg_mask)

    divided_by_seq_len = True
    trg_lens = [trg_seq_len] * batch_size
    trg_lens[batch_size - 1] = trg_seq_len - 2
    trg_lens[batch_size - 2] = trg_seq_len - 1

    coverage_attn = False

    coverage = torch.rand((batch_size, trg_seq_len, src_seq_len)) * 5
    attn_dist = torch.randint(0, 5, (batch_size, trg_seq_len, src_seq_len))
    attn_dist = F.softmax(attn_dist, dim=-1)
    lambda_coverage = 1

    loss = masked_cross_entropy(class_dist, target, trg_mask, divided_by_seq_len=divided_by_seq_len, trg_lens=trg_lens,
                         coverage_attn=coverage_attn, coverage=coverage, attn_dist=attn_dist, lambda_coverage=lambda_coverage)
    print(loss)
