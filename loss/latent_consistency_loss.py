import torch
import torch.nn as nn


class LatentConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LatentConsistencyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, real_means, real_log_vars, fake_means, fake_log_vars):

        total_lc_loss = 0.0

        for mu_real, log_var_real, mu_fake, log_var_fake in zip(real_means, real_log_vars, fake_means, fake_log_vars):
            mu_real = mu_real.detach()
            log_var_real = log_var_real.detach()

            var_real = torch.exp(log_var_real)
            var_fake = torch.exp(log_var_fake)

            # D_KL(z || z')
            kl = (mu_real - mu_fake).pow(2) / var_fake
            kl += var_real / var_fake
            kl -= 1.0
            kl -= (log_var_real - log_var_fake)

            if self.reduction == 'mean':
                kl = 0.5 * kl.mean()
            elif self.reduction == 'sum':
                kl = 0.5 * kl.sum()
            else:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")

            total_lc_loss = kl

        return total_lc_loss