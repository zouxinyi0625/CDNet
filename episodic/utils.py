import torch
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity)


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

    plt.show()

def compute_entropy(scores):
    scores = scores.data.cpu().numpy()
    nums = scores.shape[0]
    loss = 0.0
    for i in range(nums):
        loss += -(1/scores.shape[1] * np.log(scores[i])).sum()
    return loss / nums

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


