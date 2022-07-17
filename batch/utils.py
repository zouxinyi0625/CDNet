import os
import itertools
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from matplotlib.pyplot import MultipleLocator
import torch.nn.functional as F
from scipy import stats


def plot_features(features, labels, num_classes, dirname, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, feature_dim). 3589*256(tuple)
        labels: (num_instances).
    """
    prev_time = datetime.now()

    # for font in fm.fontManager.ttflist:
    #     print(font.name)
    # plt.rcParams['font.family'] = ['SimHei']  # 中文

    feat_pca = PCA(n_components=50).fit_transform(features)
    feat_pca_tsne = TSNE(n_components=2).fit_transform(feat_pca)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    for label_idx in range(num_classes):
        plt.scatter(
            feat_pca_tsne[labels == label_idx, 0],
            feat_pca_tsne[labels == label_idx, 1],
            c=colors[label_idx],
            s=4,
        )

    if dirname == 'CK+':
        plt.legend(['Angry', 'Surprise', 'Disgust', 'Fear', 'Happy', 'Sad', 'Contempt'], loc='best', markerscale=1.5)
    elif dirname in ['MMI', 'OULU']:
        plt.legend(['AN', 'SU', 'DI', 'FE', 'HA', 'SA'], loc='upper right', markerscale=2)
    elif dirname == 'RAF':
        plt.legend(['Surprise', 'Fear', 'Digust', 'Happy', 'Sad', 'Angry', 'Neutral'], loc='best', markerscale=1.5)
        # plt.legend(['惊讶', '恐惧', '厌恶', '高兴', '悲伤', '愤怒', '中性'], loc='best', markerscale=1.5)
    elif dirname == 'SFEW':
        plt.legend(['Angry', 'Digust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'], loc='best', markerscale=1.5)
    elif dirname == "multi":
        plt.legend(["neutral", "anger", "surprise", "disgust", "afraid", "happy", "sadness", "contempt"], loc='best',
                   markerscale=1.5)
    elif dirname == "primary":
        plt.legend(["p1", "p2", "p3"], loc='best', markerscale=1.5)


def plot_confusion_matrix(cm, class_names, dir,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "matrix.jpg"))


def write_results(file, txt):
    with open(file, 'a') as f:
        f.write(txt)
        f.write('\n')
    print(txt)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def flatmatrix(matrix):
    result = []
    for i in range(len(matrix)):
        result.extend(matrix[i])

    return result


def distance(A, B):
    # A [m,k] B [n,k] distmat [m,n]
    m = A.shape[0]
    n = B.shape[0]
    sumSqAEx = torch.pow(A, 2).sum(1).unsqueeze(dim=1).expand(m, n)
    sumSqBEx = (torch.pow(B, 2).sum(1)).expand(m, n)
    distmat = sumSqAEx + sumSqBEx
    distmat.addmm_(1, -2, A, B.t())

    print(distmat)


def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)

    print(ED)

    off_diag_ix = np.where(~np.eye(ED.shape[0], dtype=bool))
    ortho_reg = ED[off_diag_ix]  # select off-diagonal elements

    total_reg = np.mean(ortho_reg)
    return total_reg


def variance(matrix):
    mean_matrix = np.mean(matrix, axis=0)
    return np.mean(np.sqrt(np.power(matrix - mean_matrix, 2).sum(1)))


def cosine_similarity(a, b):
    # norm
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    # sim
    sim = (np.matmul(a, b)) / (ma * mb)
    return sim


def independence(matrix):
    images = matrix.shape[0]
    sims = np.array([0., 0., 0.])
    for i in range(images):
        a = matrix[i][0]
        b = matrix[i][1]
        c = matrix[i][2]

        sims[0] += cosine_similarity(a, b)
        sims[1] += cosine_similarity(b, c)
        sims[2] += cosine_similarity(a, c)

    return sims / images


def similarity_tensor(matrix):
    # matrix [images,color,dim]
    batch_size = matrix.shape[0]
    for i in range(batch_size):
        primary = matrix[i]
        # norm
        primary = F.normalize(primary)
        sim_matrix = primary.mm(primary.T)

    return sim_matrix


def plot_distribution(logits, i):
    colors = ['C0', 'C1', 'C2']
    N = 8
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, logits, width, color=colors[i])

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Label distribution of Primary {}'.format(str(i + 1)))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(["neutral", "anger", "surprise", "disgust", "afraid", "happy", "sadness", "contempt"])
    plt.show()


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def plot_weights(weight_exp):
    # weight_exp [num_classes, n_group]
    legends = ['surprise', "afraid", "disgust", "happy", "sadness", "anger", "neutral"]
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    for i in range(len(legends)):
        plt.plot(np.array([1, 2, 3]), weight_exp[i].data.cpu().numpy(), label=legends[i])
        plt.scatter(np.array([1, 2, 3]), weight_exp[i].data.cpu().numpy())

    ax.xaxis.set_major_locator(x_major_locator)

    plt.legend()
    plt.show()


def compute_entropy(scores):
    scores = scores.data.cpu().numpy()
    nums = scores.shape[0]
    loss = 0.0
    for i in range(nums):
        loss += -(1/scores.shape[1] * np.log(scores[i])).sum()
    return loss / nums


def entropy_loss(scores):
    scores = scores.data.cpu().numpy()
    nums = scores.shape[0]
    loss = 0.0
    for i in range(nums):
        loss += stats.entropy(scores[i])
    return loss / nums


if __name__ == "__main__":
    # matrix = np.array([[1., 2., 3.], [1., 1., 1.], [3., 4., 5.], [0., 1., 1.]])
    # matrix = torch.from_numpy(matrix)
    # dis_m = EuclideanDistances(matrix, matrix)
    # var_m = variance(matrix)
    #
    # logits = [0.5, -0.1, 0.2, 0.3, 0.4, 0.1, 0.1]
    # plot_distribution(logits)
    # print(dis_m)

    # matrix = np.random.rand(64, 3, 512)
    # ind = independence(matrix)
    #
    # matrix = torch.from_numpy(matrix)
    # ind2 = similarity_tensor(matrix)
    #
    # print(ind)
    # print(ind2)

    a = torch.randn(3, 7)
    a = F.softmax(a)
    print(entropy_loss(a))
    print(compute_entropy(a))
