import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction


    def forward(self, tangent_means_tuple, tangent_log_vars_tuple):
        kl_loss = 0.0

        for mu, log_var in zip(tangent_means_tuple, tangent_log_vars_tuple):

            kl_element = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
            if self.reduction == 'mean':
                kl_loss += kl_element.mean()
            elif self.reduction == 'sum':
                kl_loss += kl_element.sum()
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return kl_loss