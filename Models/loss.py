import torch
import torch.nn as nn
import torch.nn.functional as F

class PhononLoss(nn.Module):
    def __init__(self, mse_weight=0.7, first_order_weight=0.2, second_order_weight=0.1, eps=1e-8):
        super().__init__()
        self.mse_weight = mse_weight
        self.first_order_weight = first_order_weight
        self.second_order_weight = second_order_weight
        self.eps = eps

    def forward(self, pred_f, true_f, band_mask):
        # pred_f/true_f: [B, num_bands, num_qpoints] e.g., [B, 120, 256]
        # band_mask: [B, num_bands], 1 for valid band, 0 for padded

        # ---------- 1. Data loss (MSE on raw frequencies) ----------
        mse = F.mse_loss(pred_f, true_f, reduction='none').mean(dim=-1)  # [B, num_bands]
        data_loss = (mse * band_mask).sum() / (band_mask.sum() + self.eps)

        # Expand mask to q-point dimension for broadcasting
        band_mask_exp = band_mask.unsqueeze(-1)  # [B, num_bands, 1]

        # ---------- 2. First-order smooth loss ----------
        pred_diff1 = pred_f[..., 1:] - pred_f[..., :-1]   # [B, num_bands, L-1]
        true_diff1 = true_f[..., 1:] - true_f[..., :-1]
        mask1 = band_mask_exp.expand_as(pred_diff1)
        diff_first = (pred_diff1 - true_diff1) ** 2
        diff_norm1 = 0.5 * (pred_diff1 ** 2 + true_diff1 ** 2) + self.eps
        smooth_loss1 = (diff_first / diff_norm1 * mask1).sum() / (mask1.sum() + self.eps)

        # ---------- 3. Second-order curvature loss ----------
        pred_diff2 = pred_f[..., 2:] - 2 * pred_f[..., 1:-1] + pred_f[..., :-2]  # [B, num_bands, L-2]
        true_diff2 = true_f[..., 2:] - 2 * true_f[..., 1:-1] + true_f[..., :-2]
        mask2 = band_mask_exp.expand_as(pred_diff2)
        diff_second = (pred_diff2 - true_diff2) ** 2
        diff_norm2 = 0.5 * (pred_diff2 ** 2 + true_diff2 ** 2) + self.eps
        smooth_loss2 = (diff_second / diff_norm2 * mask2).sum() / (mask2.sum() + self.eps)

        # ---------- Total loss ----------
        total = (self.mse_weight * data_loss +
                 self.first_order_weight * smooth_loss1 +
                 self.second_order_weight * smooth_loss2)

        return total, data_loss, smooth_loss1, smooth_loss2