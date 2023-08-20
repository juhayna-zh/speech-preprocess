import torch
from itertools import permutations

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def Loss(ests, egs):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def collate_fn(batch):
    max_len = max([(d[0].shape[-1]) for d in batch])
    
    padded_batch_mix = torch.zeros(len(batch), 1, max_len)
    padded_batch_spk1 = torch.zeros(len(batch), 1, max_len)
    padded_batch_spk2 = torch.zeros(len(batch), 1, max_len)
    
    sr = batch[0][3]
    
    for i, d in enumerate(batch):
        mix, spk1, spk2, _ = d
        padded_batch_mix[i, :, :mix.shape[-1]] = mix
        padded_batch_spk1[i, :, :spk1.shape[-1]] = spk1
        padded_batch_spk2[i, :, :spk2.shape[-1]] = spk2
        
    return padded_batch_mix, padded_batch_spk1, padded_batch_spk2, sr
 