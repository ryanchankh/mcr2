import numpy as np
import torch

import train_func as tf
import utils

from itertools import combinations


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01, device="cpu"):
        super(MaximalCodingRateReduction, self).__init__()
        # pre-allocate tensors
        self.I = None
        self.trPi = None
        self.scalar = None
        self.q_exp = None
        self.log_temp = None
        self.log_det = None
        self.compress_loss = None
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device

    def compute_discrimn_loss_empirical_batch(self, W):
        """Batchwise Empirical Discriminative Loss."""
        b, p, m = W.shape
        # I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(self.I[:b,:] + self.gam1 * scalar * W.matmul(W.transpose(1,2)))
        return torch.mean(logdet) / 2.

    def compute_compress_loss_empirical_batch(self, W, q):
        """Batchwise Empirical Compressive Loss."""
        b, p, m = W.shape
        b_pi, k, m_pi = q.shape
        assert(b == b_pi)
        if (self.trPi is None):
            self.trPi = torch.zeros(b_pi, m_pi).to(self.device)
            self.scalar = torch.zeros(b_pi, m_pi).to(self.device)
            self.log_temp = torch.zeros(b, k, p, p).to(self.device)
            self.log_det = torch.zeros(b, k).to(self.device)
            self.q_exp = torch.zeros(b, k, p, m_pi).to(self.device)
            self.compress_loss = torch.zeros(b).to(self.device)
        self.trPi = torch.sum(q, axis=1) + 1e-8
        self.scalar = p / (self.trPi * self.eps)        

        self.q_exp = q.transpose(1,2).unsqueeze(2).repeat(1,1,p,1)
        self.log_temp = torch.mul(W.unsqueeze(1),self.q_exp).matmul(W.transpose(1,2).unsqueeze(1))

        self.log_det = torch.logdet(self.I.unsqueeze(1)[:b,:] + torch.mul(self.scalar.unsqueeze(2).unsqueeze(3), self.log_temp))
        self.compress_loss = torch.sum(self.log_det * self.trPi / m, axis=1)

        return torch.mean(self.compress_loss) / 2.

    def compute_discrimn_loss_theoretical_batch(self, W):
        """Batchwise Theoretical Discriminative Loss."""
        b, p, m = W.shape
        scalar = p / (m * self.eps)
        logdet = torch.logdet(self.I + scalar * W.matmul(W.transpose(1,2)))
        return torch.mean(logdet) / 2.

    def compute_compress_loss_theoretical_batch(self, W, Pi):
        """Batchwise Empirical Compressive Loss."""
        b, p, m = W.shape
        b_pi, k, _, _ = Pi.shape
        assert(b == b_pi)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.sum(Pi[:,j,:].view(b,-1), axis=1) + 1e-8
            scalar = p / (trPi * self.eps)
            log_temp = W.matmul(Pi[:,j,:]).matmul(W.transpose(1,2))
            log_det = torch.logdet(self.I + torch.mul(scalar.unsqueeze(1).unsqueeze(2), log_temp))
            compress_loss += log_det * trPi / (2 * m)
        return torch.mean(compress_loss)

    def norm_latent(self, W):
        """
        for W.size() = (b, d, m), return batch-wise normalized latents, i.e.

        $$ ||W||_F^2 = m $$
        """
        b, d, m = W.size()
        foo = torch.sqrt(torch.sum(W.reshape(W.size(0), -1)**2, axis=1).unsqueeze(1))
        bar = np.sqrt(m)*W.reshape(W.size(0), -1) / foo
        return bar.view(b, d, m)

    def forward(self, X, q):
        """
        using MCR loss in self-supervised setting
        change input is q(z,c) matrix rather than class labels (Y)
        """

        W = X.transpose(1,2) # presumably, we want to transpose the non-batch axes 

        b, p, _ = W.shape # shape?
        b_pi, k, m, = q.shape
        if self.I is None:
            self.I = torch.eye(p).unsqueeze(0).repeat(b,1,1).to(self.device)
        assert(b_pi == b)

        W = self.norm_latent(W)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical_batch(W)
        compress_loss_empi = self.compute_compress_loss_empirical_batch(W, q)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical_batch(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical_batch(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return total_loss_empi #,
                # [discrimn_loss_empi.item(), compress_loss_empi.item()])#,
                # [discrimn_loss_theo.item(), compress_loss_theo.item()])
