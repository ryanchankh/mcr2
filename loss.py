import numpy as np
import torch

import train_func as tf
import utils

from itertools import combinations


class CompressibleLoss(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(CompressibleLoss, self).__init__()
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
            compress_loss += log_det
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


class CompressibleLoss2(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLoss2, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 2)))
        num_pairs = int(1.0 * (num_imgs * (num_imgs + 1)) / 2.0)
        sample_idx = np.random.choice(len(pair_combs), num_pairs)
        sample_pairs = pair_combs#[sample_idx]
        compress_loss_ortho = 0.
        for step, (i, j) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi = np.diag(Pi)
            Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            compress_loss_ortho += log_det
        return compress_loss_ortho

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
        Pi = tf.label_to_membership(y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        compress_loss_empi_ortho = self.compute_compress_loss_empirical_ortho(W, y)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],
                compress_loss_empi_ortho.item())


class CompressibleLoss3(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLoss3, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 2)))
        # num_pairs = int(1.0 * (num_imgs * (num_imgs + 1)) / 2.0)
        num_pairs = 500
        sample_idx = np.random.choice(len(pair_combs), num_pairs)
        sample_pairs = pair_combs[sample_idx]
        compress_loss_ortho = 0.
        loss_c, loss_d = 0, 0
        for step, (i, j) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi = np.diag(Pi)
            Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            compress_loss_ortho += log_det
            loss_d += log_det

            Pi_i = np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_i = np.diag(Pi_i)
            Pi_i = torch.tensor(Pi_i, dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))
            compress_loss_ortho -= 0.5 * log_det_i
            loss_c += log_det_i.detach()

            Pi_j = np.zeros(len(y))
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_j = np.diag(Pi_j)
            Pi_j = torch.tensor(Pi_j, dtype=torch.float32).cuda()
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_j = p / (trPi_j * self.eps)
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))
            compress_loss_ortho -= 0.5 * log_det_j
            loss_c += log_det_j.detach()

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
        compress_loss_empi_ortho, discrimn_loss_empi, compress_loss_empi = self.compute_compress_loss_empirical_ortho(W, y)
        discrimn_loss_theo = 0 #self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = 0 #self.compute_compress_loss_theoretical(W, Pi)
        # total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
        total_loss_empi = self.gam3 * -compress_loss_empi_ortho
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo, compress_loss_theo],
                compress_loss_empi_ortho.item())


class CompressibleLossContrastive(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLossContrastive, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 2)))
        # sample_idx = np.random.choice(len(pair_combs), num_pairs, replace=False)
        sample_pairs = pair_combs
        R, dR = [], []
        for step, (i, j) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi = np.diag(Pi)
            Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            loss_d = log_det

            Pi_i = np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_i = np.diag(Pi_i)
            Pi_i = torch.tensor(Pi_i, dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))

            Pi_j = np.zeros(len(y))
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_j = np.diag(Pi_j)
            Pi_j = torch.tensor(Pi_j, dtype=torch.float32).cuda()
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_j = p / (trPi_j * self.eps)
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))

            loss_c = 0.5 * (log_det_i + log_det_j)
            R.append(loss_d)
            dr = loss_d - loss_c
            dR.append(dr.cpu().detach().item())
        idx = np.argsort(dR)
        return R[idx][-140:].mean()

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
        Pi = tf.label_to_membership(y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        compress_loss_empi_ortho = self.compute_compress_loss_empirical_ortho(W, y)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],
                compress_loss_empi_ortho.item())


class CompressibleLossCompressive(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLossCompressive, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 2)))
        # sample_idx = np.random.choice(len(pair_combs), num_pairs, replace=False)
        sample_pairs = pair_combs
        R, dR = [], []
        for step, (i, j) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi = np.diag(Pi)
            Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            loss_d = log_det

            Pi_i = np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_i = np.diag(Pi_i)
            Pi_i = torch.tensor(Pi_i, dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))

            Pi_j = np.zeros(len(y))
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_j = np.diag(Pi_j)
            Pi_j = torch.tensor(Pi_j, dtype=torch.float32).cuda()
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_j = p / (trPi_j * self.eps)
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))

            loss_c = 0.5 * (log_det_i + log_det_j)
            R.append(loss_d)
            dr = loss_d - loss_c
            dR.append(dr.cpu().detach().item())
        idx = np.argsort(dR)
        return R[idx][-140:].mean()

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
        Pi = tf.label_to_membership(y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        compress_loss_empi_ortho = self.compute_compress_loss_empirical_ortho(W, y)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],
                compress_loss_empi_ortho.item())



class CompressibleLossTriplet(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLoss4, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 3)))
        # sample_idx = np.random.choice(len(pair_combs), num_pairs, replace=False)
        sample_pairs = pair_combs
        R, dR = [], []
        for step, (i, j, k) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi[k * self.num_aug:(k + 1) * self.num_aug] = 1.
            Pi = torch.tensor(np.diag(Pi), dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            loss_d = log_det

            Pi_i = np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_i = torch.tensor(np.diag(Pi_i), dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))

            Pi_j = np.zeros(len(y))
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_j = torch.tensor(np.diag(Pi_j), dtype=torch.float32).cuda()
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_j = p / (trPi_j * self.eps)
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))

            Pi_k = np.zeros(len(y))
            Pi_k[k * self.num_aug:(k + 1) * self.num_aug] = 1.
            Pi_k = torch.tensor(np.diag(Pi_k), dtype=torch.float32).cuda()
            trPi_k = torch.trace(Pi_k) + 1e-8
            scalar_k = p / (trPi_k * self.eps)
            log_det_k = torch.logdet(I + scalar_k * W.matmul(Pi_k).matmul(W.T))

            loss_c = 0.5 * (log_det_i + log_det_j + log_det_k)
            R.append(loss_d)
            dr = loss_d - loss_c
            dR.append(dr.cpu().detach().item())
        idx = np.argsort(dR)
        return R[idx][-140:].mean()

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
        Pi = tf.label_to_membership(y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        compress_loss_empi_ortho = self.compute_compress_loss_empirical_ortho(W, y)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],
                compress_loss_empi_ortho.item())


class CompressibleLossPoly(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, gam3=0.01, eps=0.01, num_aug=10):
        super(CompressibleLossPoly, self).__init__()
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

    def compute_compress_loss_empirical_ortho(self, W, y):
        """Empirical Compressive Loss (ortho)."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        num_imgs = len(y) // self.num_aug
        pair_combs = np.array(list(combinations(range(num_imgs), 3)))
        # sample_idx = np.random.choice(len(pair_combs), num_pairs, replace=False)
        sample_pairs = pair_combs
        R, dR = [], []
        for step, (i, j, k) in enumerate(sample_pairs):
            Pi = np.zeros(len(y))
            Pi[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi[k * self.num_aug:(k + 1) * self.num_aug] = 1.
            Pi = torch.tensor(np.diag(Pi), dtype=torch.float32).cuda()
            trPi = torch.trace(Pi) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi).matmul(W.T))
            loss_d = log_det

            Pi_i = np.zeros(len(y))
            Pi_i[i * self.num_aug:(i + 1) * self.num_aug] = 1.
            Pi_i = torch.tensor(np.diag(Pi_i), dtype=torch.float32).cuda()
            trPi_i = torch.trace(Pi_i) + 1e-8
            scalar_i = p / (trPi_i * self.eps)
            log_det_i = torch.logdet(I + scalar_i * W.matmul(Pi_i).matmul(W.T))

            Pi_j = np.zeros(len(y))
            Pi_j[j * self.num_aug:(j + 1) * self.num_aug] = 1.
            Pi_j = torch.tensor(np.diag(Pi_j), dtype=torch.float32).cuda()
            trPi_j = torch.trace(Pi_j) + 1e-8
            scalar_j = p / (trPi_j * self.eps)
            log_det_j = torch.logdet(I + scalar_j * W.matmul(Pi_j).matmul(W.T))

            Pi_k = np.zeros(len(y))
            Pi_k[k * self.num_aug:(k + 1) * self.num_aug] = 1.
            Pi_k = torch.tensor(np.diag(Pi_k), dtype=torch.float32).cuda()
            trPi_k = torch.trace(Pi_k) + 1e-8
            scalar_k = p / (trPi_k * self.eps)
            log_det_k = torch.logdet(I + scalar_k * W.matmul(Pi_k).matmul(W.T))

            loss_c = 0.5 * (log_det_i + log_det_j + log_det_k)
            R.append(loss_d)
            dr = loss_d - loss_c
            dR.append(dr.cpu().detach().item())
        idx = np.argsort(dR)
        return R[idx][-140:].mean()

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
        num_samples = [15, 10, 5, 3]
        loss = torch.tensor(0., dtype=torch.float32).cuda()
        for num_sample in range(num_samples):
            sample_idx = np.random.choice(len(y) // self.num_aug, num_sample)
            sample_y = [np.repeat(i, self.num_aug) for i in range(num_sample)]
            sample_W = X[sample_idx].T
            sample_Pi = tf.label_to_membership(np.array(sample_y))
            sample_Pi = torch.tensor(sample_Pi, dtype=torch.float32).cuda()
            discrimn_loss_empi = self.compute_discrimn_loss_empirical(sample_W)
            compress_loss_empi = self.compute_compress_loss_empirical(sample_W, sample_Pi)
            discrimn_loss_theo = self.compute_discrimn_loss_theoretical(sample_W)
            compress_loss_theo = self.compute_compress_loss_theoretical(sample_W, sample_Pi)
            total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi - self.gam3 * compress_loss_empi_ortho
            loss += total_loss_empi / num_sample
    
        return (loss,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()],
                compress_loss_empi_ortho.item())