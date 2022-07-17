import numpy as np
import torch.nn.functional as F
from scipy import stats
import torch


def cosine_similarity(a, b):
    # norm
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # sim
    sim = (np.matmul(a, b)) / (ma * mb)
    return sim


def orthloss(primarys):
    primarys = primarys.transpose(0, 1)

    batch_size = primarys.shape[0]

    loss = 0.0

    for i in range(batch_size):
        primary = primarys[i]

        sim_matrix = primary.mm(primary.T)

        # 取非对角线元素
        off_diag_ix = np.where(~np.eye(sim_matrix.shape[0], dtype=bool))
        sims = sim_matrix[off_diag_ix]  # select off-diagonal elements

        loss += sims.mean()

    return loss / batch_size


def entropy_loss(scores):
    scores = scores.data.cpu().numpy()
    nums = scores.shape[0]
    loss = 0.0
    for i in range(nums):
        loss += stats.entropy(scores[i])
    return loss / nums

