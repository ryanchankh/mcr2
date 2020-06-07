import numpy as np
import torch

import train_func as tf
import utils

from itertools import combinations


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = tf.label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])


class MaximalCodingRateReductionPair(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(MaximalCodingRateReductionPair, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.eps = eps
        self.num_aug = num_aug

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det
        return compress_loss

    def compute_pair_loss(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        sample_pairs = np.array(list(combinations(range(num_imgs), 2)))
        compress_loss_ortho = 0.
        loss_c, loss_d = 0, 0
        for step, (i, j) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi = torch.tensor(np.diag(Pi), dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            compress_loss_ortho += self.gam2 * log_det
            loss_d += log_det

            Pi_i, Pi_j = np.zeros(len(y)), np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_i = torch.tensor(np.diag(Pi_i), dtype=torch.float32).cuda()
            Pi_j = torch.tensor(np.diag(Pi_j), dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            scalar_j = p / (trPi_j * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))
            compress_loss_ortho -= log_det_j + log_det_i

            loss_d += log_det.detach()
            loss_c += log_det_i.detach() + log_det_j.detach()

        return (compress_loss_ortho / len(sample_pairs),
                loss_d / len(sample_pairs),
                loss_c / len(sample_pairs))

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, y, num_classes=None):
        if num_classes is None:
            num_classes = y.max() + 1
        W = X.T
        pair_loss, disrimn_loss, compress_loss = self.compute_pair_loss(W, y)
        total_loss_empi = self.gam3 * -pair_loss
        return (total_loss_empi, 
                discrimn_loss.item(), 
                compress_loss.item(), 
                pair_loss.item())