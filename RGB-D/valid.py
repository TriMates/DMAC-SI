# import numpy as np
# a=np.arange(10)
# print(a)
# index=0
# index_array = np.arange(10)
# for i in range(20):
#     idx = index_array[index : min((index + 1) , 10)]
#     print(a[idx])
#
#     index = index + 1 if (index + 1)  <= 10 else 0
#     print(a[index])
class MFeatDataSet():
    '''Mixed-modal feature'''
    from __future__ import print_function, absolute_import, division

    import os
    import pickle
    import pdb

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import scipy.io as sio
    from PIL import Image
    from torch.utils.data import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn import metrics
    from sklearn.metrics.cluster.supervised import contingency_matrix
    from munkres import Munkres
    import math

    import numpy as np
    from sklearn.utils import shuffle
    from sklearn.preprocessing import MinMaxScaler
    import scipy.io as sio
    def __init__(self):
        self.file_mat = sio.loadmat('data/bdgp/bdgp.mat')
        self.lens = 2500
        data = self.file_mat['X']
        feat0 = self.file_mat['Ya']
        feat0 = feat0.astype(np.float32)
        cluster_label = self.file_mat['truthF']
        cluster_label = cluster_label.astype(np.float32).squeeze()
        modality0 = sio.loadmat('data/bdgp/bdgp_0.3_m0.mat')['array'].astype(np.float32)
        modality1 = sio.loadmat('data/bdgp/bdgp_0.3_m1.mat')['array'].astype(np.float32)
        modality2 = sio.loadmat('data/bdgp/bdgp_0.3_m2.mat')['array'].astype(np.float32)
        modality3 = sio.loadmat('data/bdgp/bdgp_0.3_m3.mat')['array'].astype(np.float32)

        data, self.feat0, self.cluster_label, self.modality0, self.modality1, self.modality2, self.modality3 = shuffle(
            data, feat0, cluster_label, modality0, modality1, modality2, modality3)
        self.feat1 = data[:, 0:1000].astype(np.float32)
        self.feat2 = data[:, 1000:1500].astype(np.float32)
        self.feat3 = data[:, 1500:1750].astype(np.float32)

    def __getitem__(self, index):
        feat0 = self.feat0[index]
        feat1 = self.feat1[index]
        feat2 = self.feat2[index]
        feat3 = self.feat3[index]
        cluster_label = self.cluster_label[index]
        modality0 = self.modality0[index]
        modality1 = self.modality1[index]
        modality2 = self.modality2[index]
        modality3 = self.modality3[index]

        return np.float32(index), feat0, feat1, feat2, feat3, modality0, modality1, modality2, modality3, cluster_label

    def __len__(self):
        return self.lens