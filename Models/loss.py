import torch
import torch.nn as nn
import torch.nn.functional as F

class PhononLoss(nn.Module):
    def __init__(self, mse_weight=0.8, smooth_weight=0.2, eps=1e-8):
        super().__init__()
        self.mse_weight = mse_weight
        self.smooth_weight = smooth_weight
        self.eps = eps

    def forward(self, pred_f, true_f, band_mask):
        # pred_f/true_f: [B, 120, 256], band_mask: [B, 120]
        # data_loss
        mse = F.mse_loss(pred_f, true_f, reduction='none').mean(dim=-1)  # [B,120]
        data_loss = (mse * band_mask).sum() / (band_mask.sum() + self.eps)

        # smooth_loss: 对相邻q点的差分做MSE
        pred_diff = pred_f[..., 1:] - pred_f[..., :-1]   # [B,120,255]
        true_diff = true_f[..., 1:] - true_f[..., :-1]
        valid_t = band_mask.unsqueeze(-1)                # [B,120,1]
        smooth_mask = valid_t.expand_as(pred_diff)
        
        diff_error = (pred_diff - true_diff) ** 2
        diff_norm = 0.5 * (pred_diff ** 2 + true_diff ** 2) + self.eps
        smooth_loss = (diff_error / diff_norm * smooth_mask).sum() / (smooth_mask.sum() + self.eps)

        total = self.mse_weight * data_loss + self.smooth_weight * smooth_loss
        return total, data_loss, smooth_loss