from time import time
from numpy import hstack
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import scipy.io as sio
import platform
from sklearn.metrics import log_loss
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Flatten, Reshape, Conv3D, Conv3DTranspose, MaxPooling2D, Dropout, GlobalMaxPooling2D
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Multiply, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import Regularizer, l1, l2, l1_l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, SparsePCA
from math import log
import Nmetrics
import matplotlib.pyplot as plt


def FAE(dims, act='relu', view=1):
    n_stacks = len(dims) - 1
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')

    input_name = 'v'+str(view)+'_'
    # input
    x = Input(shape=(dims[0],), name='input' + str(view))

    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name=input_name+'encoder_%d' % i)(h)
 
    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='embedding' + str(view))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name=input_name+'decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name=input_name+'decoder_0')(y)

    return Model(inputs=x, outputs=y, name=input_name+'Fae'), Model(inputs=x, outputs=h, name=input_name+'Fencoder')

def MAE(view=2, filters=[32, 64, 128, 10], view_shape = [1, 2, 3]):
    # print(len(view_shape[0]))
    if len(view_shape[0]) == 1:
        typenet = 'f-f'          # Fully connected networks
    else:
        typenet = 'c-c'          # Convolution networks

    if typenet == 'c-c':
        input1_shape = view_shape[0]
        input2_shape = view_shape[1]
        if input1_shape[0] % 8 == 0:
            pad1 = 'same'
        else:
            pad1 = 'valid'
        print("----------------------")
        print(filters)
        input1 = Input(input1_shape, name='input1')
        x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v1')(input1)
        x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v1')(x)
        x = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v1')(x)
        x = Flatten(name='Flatten1')(x)
        x1 = Dense(units=filters[3], name='embedding1')(x)
        x = Dense(units=filters[2]*int(input1_shape[0]/8)*int(input1_shape[0]/8), activation='relu',
                  name='Dense1')(x1)
        x = Reshape((int(input1_shape[0]/8), int(input1_shape[0]/8), filters[2]), name='Reshape1')(x)
        x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v1')(x)
        x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v1')(x)
        x = Conv2DTranspose(input1_shape[2], 5, strides=2, padding='same', name='deconv1_v1')(x)

        input2 = Input(input2_shape, name='input2')
        xn = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v2')(input2)
        xn = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v2')(xn)
        xn = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v2')(xn)
        xn = Flatten(name='Flatten2')(xn)
        x2 = Dense(units=filters[3], name='embedding2')(xn)
        xn = Dense(units=filters[2] * int(input2_shape[0] / 8) * int(input2_shape[0] / 8), activation='relu',
                   name='Dense2')(x2)
        xn = Reshape((int(input2_shape[0] / 8), int(input2_shape[0] / 8), filters[2]), name='Reshape2')(xn)
        xn = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v2')(xn)
        xn = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v2')(xn)
        xn = Conv2DTranspose(input2_shape[2], 5, strides=2, padding='same', name='deconv1_v2')(xn)
        encoder1 = Model(inputs=input1, outputs=x1)
        encoder2 = Model(inputs=input2, outputs=x2)
        ae1 = Model(inputs=input1, outputs=x)
        ae2 = Model(inputs=input2, outputs=xn)

        if view == 2:
            return [ae1, ae2], [encoder1, encoder2]
        else:
            input3_shape = view_shape[2]
            input3 = Input(input3_shape, name='input3')
            xr = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1_v3')(input3)
            xr = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2_v3')(xr)
            xr = Conv2D(filters[2], 3, strides=2, padding=pad1, activation='relu', name='conv3_v3')(xr)
            xr = Flatten(name='Flatten3')(xr)
            x3 = Dense(units=filters[3], name='embedding3')(xr)
            xr = Dense(units=filters[2] * int(input3_shape[0] / 8) * int(input3_shape[0] / 8), activation='relu',
                       name='Dense3')(x3)
            xr = Reshape((int(input3_shape[0] / 8), int(input3_shape[0] / 8), filters[2]), name='Reshape3')(xr)
            xr = Conv2DTranspose(filters[1], 3, strides=2, padding=pad1, activation='relu', name='deconv3_v3')(xr)
            xr = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2_v3')(xr)
            xr = Conv2DTranspose(input2_shape[2], 5, strides=2, padding='same', name='deconv1_v3')(xr)

            encoder3 = Model(inputs=input3, outputs=x3)
            ae3 = Model(inputs=input3, outputs=xr)

            return [ae1, ae2, ae3], [encoder1, encoder2, encoder3]

    if typenet == 'f-f':
        ae = []
        encoder = []

        for v in range(view):
            ae_tmp, encoder_tmp = FAE(dims=[view_shape[v][0], 500, 500, 2000, 10], view=v + 1)
            ae.append(ae_tmp)
            encoder.append(encoder_tmp)

        return ae, encoder

class RCLayer_softmax(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RCLayer_softmax, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.softmax(q, axis=1)

        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], )

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(RCLayer_softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2    
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MvDEC(object):
    def __init__(self,
                 filters=[32, 64, 128, 10], 

                 n_clusters=10,
                 alpha=1.0, view_shape = [1, 2, 3, 4, 5, 6]):

        super(MvDEC, self).__init__()

        self.view_shape = view_shape
        self.filters = filters
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        # prepare MvDEC model
        self.view = len(view_shape)
        # print(len(view_shape))

        self.AEs, self.encoders = MAE(view=self.view, filters=self.filters, view_shape=self.view_shape)
        Input=[]
        Output = []
        Input_e = []
        Output_e = []
        clustering_layer = []
        reclustering_layer = []
        dis_valid=[]
        self.dis = []
        # 鉴别器
        for v in range(self.view):
                dis_tem = self.discriminators(self.view_shape[v][0])
                dis_tem.compile(loss='binary_crossentropy',loss_weights=[0.001],optimizer=Adam(0.00005, 0.9))
                dis_tem.trainable= False
                self.dis.append(dis_tem)
        for v in range(self.view):
                Input.append(self.AEs[v].input)
                Output.append(self.AEs[v].output)
                Input_e.append(self.encoders[v].input)
                Output_e.append(self.encoders[v].output)
                clustering_layer.append(ClusteringLayer(self.n_clusters, name='clustering'+str(v+1))(self.encoders[v].output))
                dis_valid.append(self.dis[v](Output[v]))
        if self.view==2:
          z = K.concatenate([Output_e[0], Output_e[1]], axis=1)
        if self.view==6:
          z = K.concatenate([Output_e[0], Output_e[1],Output_e[2], Output_e[3],Output_e[4], Output_e[5]], axis=1)
        reclustering_layer.append(RCLayer_softmax(self.n_clusters,name='rcclustering')(z))
        self.autoencoder = Model(inputs=Input, outputs=Output)    # xin _ xout
        self.encoder = Model(inputs=Input_e, outputs=Output_e)   # xin _ q
        self.rc=Model(inputs=Input,outputs=reclustering_layer)
        # 镜像
        # for v in range(self.view):
        #         Input.append(self.AEs[v].input)
        #         Output.append(self.AEs[v].output)
        #         Input_e.append(self.encoders[v].input)
        #         Output_e.append(self.encoders[v].output)
        #         clustering_layer.append(ClusteringLayer(self.n_clusters, name='clustering'+str(v+1))(self.encoders[v].output))
        #         dis_valid.append(self.dis[v](Output[v]))
        # z_rec = K.concatenate([Output_e[0], Output_e[1]], axis=1)
        # z_rec1=K.concatenate([Output_e[0], Output_e[1]], axis=1)
        # reclustering_layer.append(RCLayer_softmax(self.n_clusters,name='reclustering')(z_rec[self.view:self.view*2]))
        # reclustering_layer.append(RCLayer_softmax(self.n_clusters,name='reclustering')(z_rec1[self.view:self.view * 2]))
        Output_m = []
        for v in range(self.view):
            Output_m.append(clustering_layer[v])
            Output_m.append(Output[v])
            Output_m.append(dis_valid[v])

          #  if v== self.view*2-1:
        Output_m.append(reclustering_layer[0])
        Output_m.append(reclustering_layer[0])
        self.model = Model(inputs=Input, outputs=Output_m)   # xin _ q _ xout， vslidity，

    def discriminators(self, dims):

        model = Sequential()
        model.add(Dense(512, input_dim=dims))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        dec_data = Input(shape=(dims,))
        validity = model(dec_data)

        return Model(dec_data, validity)
    def load_weights(self, weights):  # load weights of models
        self.model.load_weights(weights)

    def predict_label(self, x):  # predict cluster labels using the output of clustering layer
        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view+1): x[view]})
        Q_and_X = self.model.predict(input_dic, verbose=0)
        y_pred = []
        for view in range(len(x)):
            # print(view)
            y_pred.append(Q_and_X[view*3].argmax(1))
        
        y_q = Q_and_X[(len(x)-1)*3]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view*3]

        # y_q = y_q/len(x)
        y_mean_pred = y_q.argmax(1)
        return y_pred, self.rc.predict(input_dic).argmax(1)

    def RC_target_distribution(self,q):
        # q.max
        #qmax = np.max(q,1)
        qmax = q.max()
        valid = np.ones(1)
        fake = np.zeros(1)

        seed = np.random.rand(1)
        #print(qmax > seed)
        out = np.where(qmax > seed, valid, fake)

        return out
    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        # return q
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss=['kld', 'mse','binary_crossentropy'], loss_weights=[0.1, 1.0,0.001]):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def train_on_batch(self, xin, yout, sample_weight=None):
        return self.model.train_on_batch(xin, yout, sample_weight)

    # SDMVC
    def new_fit(self, arg, x, y, maxiter=7e6, batch_size=1, tol=1e-3,
            UpdateCoo=200, save_dir='./results/tmp'):
        print('Begin clustering:', '-' * 60)
        print('Update Coo:', UpdateCoo)
        save_interval = int(maxiter)  # only save the initial and final model
        print('Save interval', save_interval)
        # Step 1: initialize cluster centers using k-means
        t1 = time()
        ting = time() - t1
        time_record = []
        time_record.append(int(ting))
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        rckmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        input_dic = {}
        for view in range(len(x)):
            input_dic.update({'input' + str(view + 1): x[view]})
        features = self.encoder.predict(input_dic)
        z=np.hstack(features)
        data_rec = self.autoencoder.predict(input_dic)
        features_rec = self.encoder.predict(data_rec)
        if len(x) <= 3:      # small trick: less view, more times to over arg.AR, so as to get high Aligned Rate
            arg.ARtime = 3
        else:
            arg.ARtime = 1

        y_pred = []
        center = []
        y_pred_rc = []
        center_rc = []
        rckmeans.fit_predict(z)

        center_rc.append([rckmeans.cluster_centers_])
       # self.model.get_layer(name='rcclustering').set_weights([rckmeans.cluster_centers_])

        from numpy import hstack
        from sklearn import preprocessing
        from sklearn.preprocessing import StandardScaler
        StSc = StandardScaler()
        min_max_scaler = preprocessing.MinMaxScaler()
        # --------------------------------------------
        c = 1
        if c == 1:
            for view in range(len(x)):
                y_pred.append(kmeans.fit_predict(features[view]))
                center.append([kmeans.cluster_centers_])

        elif c == 2:
            n_features = []
            for view in range(len(x)):
                # n_features.append(min_max_scaler.fit_transform(features[view]))
                n_features.append(features[view])
            z = hstack(n_features)
            print(features[0].shape, len(x), z.shape)
            y_pred.append(kmeans.fit_predict(z))
            for view in range(len(x) - 1):
                y_pred.append(y_pred[0])
            print(kmeans.cluster_centers_.shape)
            centers = kmeans.cluster_centers_
            # print(self.new_P(z, centers))
            new_P = self.new_P(z, centers)
            print(new_P.argmax(1))
            print(y_pred[0])
            for view in range(len(x)):
                b = 10 * view
                e = b + 10
                np.save('TC' + str(view + 1) + '.npy', [centers[:, b:e]])
                center.append(np.load('TC' + str(view + 1) + '.npy'))
        else:
            for view in range(len(x)):
                y_pred.append(kmeans.fit_predict(features[view]))
            print("random")
        # --------------------------------------------

        for view in range(len(x)):
            acc = np.round(Nmetrics.acc(y, y_pred[view]), 5)
            nmi = np.round(Nmetrics.nmi(y, y_pred[view]), 5)
            vmea = np.round(Nmetrics.vmeasure(y, y_pred[view]), 5)
            ari = np.round(Nmetrics.ari(y, y_pred[view]), 5)
            print('Start-' + str(view + 1) + ': acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f' % (acc, nmi, vmea, ari))

        y_pred_last = []
        y_pred_sp = []
        for view in range(len(x)):
            y_pred_last.append(y_pred[view])
            y_pred_sp.append(y_pred[view])

        # for view in range(len(x)):
        #     # break
        #     if arg.K12q == 0:
        #         self.model.get_layer(name='clustering' + str(view + 1)).set_weights(center[view])
        #     else:
        #         self.model.get_layer(name='clustering' + str(view + 1)).set_weights(center[arg.K12q - 1])

        # Step 2: deep clustering
        # logging file
        import csv
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'nmi', 'vmea', 'ari', 'loss'])
        logwriter.writeheader()

        index_array = np.arange(x[0].shape[0])
        index = 0

        Loss = []
        avg_loss = []
        kl_loss = []
        for view in range(len(x)):
            Loss.append(0)
            avg_loss.append(0)
            kl_loss.append(100000)

        flag = 1

        vf = arg.view_first

        update_interval = arg.UpdateCoo

        center_init = 0

        alignment = 0

        alignment_large = 0

        ACC = []
        NMI = []
        ARI = []
        vACC = []
        vNMI = []
        vARI = []
        Rate = []
        MVKLLoss = []
        ite = 0
        ite_epoch=[]
        all_loss=[]
        best=0.1
        while True:
            if ite % update_interval == 0:
                print('\n')
                for view in range(len(x)):
                    avg_loss[view] = Loss[view] / update_interval
                    kl_loss[view] = kl_loss[view] / update_interval

                Q_and_X = self.model.predict(input_dic)
                for view in range(len(x)):
                    y_pred_sp[view] = Q_and_X[view * 3].argmax(1)

                features = self.encoder.predict(input_dic)
                data_rec = self.autoencoder.predict(input_dic)
                features_rec = self.encoder.predict(data_rec)
                if len(x) >= 6:
                    scaler = 1
                else:
                    scaler = 0
                # print("scaler ? :"+str(scaler))
                # if the number of views is too many (such as, >= 6), we can scale the features to [0,1] to build global features
                if scaler == 1:
                    n_features = []
                    for view in range(len(x)):
                        n_features.append(min_max_scaler.fit_transform(features[view]))
                    # print(n_features[0][0])
                    # print(n_features[1][0])
                    z = hstack(n_features)
                else:
                    z = hstack(features)
                # 镜像
                if len(x) >= 6:
                    scaler = 1
                else:
                    scaler = 0
                if scaler == 1:
                    n_features = []
                    for view in range(len(x)):
                        n_features.append(min_max_scaler.fit_transform(features_rec[view]))

                    z_rec = hstack(n_features)
                else:
                    z_rec = hstack(features_rec)

                usepca = 0
                if usepca == 1:
                    pca = PCA(n_components=z.shape[1])
                    # pca = SparsePCA(n_components=60)
                    z = pca.fit_transform(z)
                    lamb = pca.explained_variance_ratio_
                    i_9 = 0
                    for i in range(z.shape[1]):
                        if np.sum(lamb[0:i+1]) > 0.95:
                            i_9 = i + 1
                            break
                    # print(lamb)
                    # z = z[:, 0:i_9]
                    # print(z.shape)

                # print(features[0].shape, len(x), z.shape)

                kmean1 = KMeans(n_clusters=self.n_clusters, n_init=100,random_state=1)
                kmean2 = KMeans(n_clusters=self.n_clusters, n_init=100)

                y_pred = kmean1.fit_predict(z)
                y_pred2 = kmean2.fit_predict(z) # k-means on global features
                # 镜像

                RC_q = self.rc.predict(input_dic).argmax(1)
                acc = np.round(Nmetrics.acc(y, y_pred), 5)
                nmi = np.round(Nmetrics.nmi(y, y_pred), 5)
                vmea = np.round(Nmetrics.vmeasure(y, y_pred), 5)
                ari = np.round(Nmetrics.ari(y, y_pred), 5)
                print('ACC=%.5f, NMI=%.5f, V-Measure=%.5f, ARI=%.5f' % (acc, nmi, vmea, ari))
                acc1 = np.round(Nmetrics.acc(y, RC_q), 5)
                nmi1 = np.round(Nmetrics.nmi(y, RC_q), 5)
                vmea1 = np.round(Nmetrics.vmeasure(y, RC_q), 5)
                ari1 = np.round(Nmetrics.ari(y, RC_q), 5)
                print('rc', 'ACC=%.5f, NMI=%.5f, V-Measure=%.5f, ARI=%.5f' % (acc1, nmi1, vmea1, ari1))


                ACC.append(acc)
                NMI.append(nmi)
                ARI.append(ari)

                if ite >= 50 :
                    print('saving model to:', save_dir + '/model_final.h5' + str(ite))
                    self.model.save_weights(save_dir +'/'+ str(ite)+ 'model_final.h5')
                # if acc>=0.73:
                #    print(acc)
                #    break
                the = np.sum(kl_loss) / len(x)
                # print(kl_loss)
                # print(Loss)
                # print(np.sum(kl_loss), np.sum(Loss))

                if y is not None:

                    scale = len(y)
                    for i in range(len(y)):
                        predict = y_pred_sp[0][i]
                        for view in range(len(x) - 1):
                            if predict == y_pred_sp[view + 1][i]:
                                continue
                            else:
                                scale -= 1
                                break

                    alignment_before = alignment
                    alignment = (scale / len(y))
                    print('aligned ratio: %.2f%%. %d' % (alignment * 100, len(y)))
                    Rate.append(alignment)
                    tmpACC = []
                    tmpNMI = []
                    tmpARI = []
                    for view in range(len(x)):
                        acc = np.round(Nmetrics.acc(y, y_pred_sp[view]), 5)
                        nmi = np.round(Nmetrics.nmi(y, y_pred_sp[view]), 5)
                        vme = np.round(Nmetrics.vmeasure(y, y_pred_sp[view]), 5)
                        ari = np.round(Nmetrics.ari(y, y_pred_sp[view]), 5)
                        logdict = dict(iter=ite, nmi=nmi, vmea=vme, ari=ari, loss=avg_loss[view])
                        logwriter.writerow(logdict)
                        logfile.flush()
                        print('V' + str(
                            view + 1) + '-Iter %d: acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f; loss=%.5f' % (
                                  ite, acc, nmi, vme, ari, avg_loss[view]))
                        tmpACC.append(acc)
                        tmpNMI.append(nmi)
                        tmpARI.append(ari)
                    vACC.append(tmpACC)
                    vNMI.append(tmpNMI)
                    vARI.append(tmpARI)
                    ting = time() - t1

                if alignment > arg.AR:
                    alignment_large += 1
                else:
                    alignment_large = 0

                if alignment_large < arg.ARtime:
                    # RC_q = self.rc.predict(input_dic)
                    # RC_Q.append(RC_q )
                    # RC_p =self.RC_target_distribution(RC_q )
                    #RC_p_rec = self.RC_target_distribution(RCLayer_softmax(self.n_clusters, name='reclustering')(z_rec))
                    Center_init = kmean1.cluster_centers_    # k-means on global features
                    new_P = self.new_P(z, Center_init)      # similarity measure
                    print("Multi-view discriminative representation learning")
                    p = self.target_distribution(new_P)
                    #镜像
                    Center_init1 = kmean2.cluster_centers_  # k-means on global features
                    new_P_rec = self.new_P(z_rec, Center_init1)  # similarity measure
                    print("Multi-view discriminative representation learning")
                    p_rec = self.target_distribution(new_P_rec)
                    # enhance discrimination
                    center_init += 1
                else:
                    break

                flag += 1
                # print('Next corresponding: p' + str(q_index))

                P = []
                P_rec = []


                # RC_P.append(RC_p)
                # RC_P_rec.append(RC_p_rec)
                if arg.Coo == 1:
                    print("unify target distribution of KL loss")
                    for view in range(len(x)):
                        P.append(p)
                        P_rec.append(p_rec)

                else:
                    print("self clustering")
                    for view in range(len(x)):
                        P.append(self.target_distribution(Q_and_X[view * 2]))

                ge = np.random.randint(0, x[0].shape[0], 1, dtype=int)
                ge = int(ge)
                # print('Number of sample:' + str(ge))
                # for view in range(len(x)):
                #     for i in Q_and_X[view * 2][ge]:
                #         print("%.3f  " % i, end="")
                #     print("\n")

                # evaluate the clustering performance
                for view in range(len(x)):
                    Loss[view] = 0.
                    kl_loss[view] = 0.
            if ite % 10== 0:
                print('print:', save_dir + '/' + str(ite))
                RC_q = self.rc.predict(input_dic).argmax(1)
                acc1 = np.round(Nmetrics.acc(y, RC_q), 5)
                nmi1 = np.round(Nmetrics.nmi(y, RC_q), 5)
                vmea1 = np.round(Nmetrics.vmeasure(y, RC_q), 5)
                ari1 = np.round(Nmetrics.ari(y, RC_q), 5)
                print('rc', 'ACC=%.5f, NMI=%.5f, V-Measure=%.5f, ARI=%.5f' % (acc1, nmi1, vmea1, ari1))
                if acc1> best:
                    best=acc1
                    print('best', 'ACC=%.5f, NMI=%.5f, V-Measure=%.5f, ARI=%.5f' % (best, nmi1, vmea1, ari1))
                    print('saving model to:', save_dir + '/model_final.h5' + str(ite))
                    self.model.save_weights(save_dir + '/' + str(ite) + 'model_final.h5')

            if ite % 200 == 0:
                RC_q = self.rc.predict(input_dic).argmax(1)
                acc1 = np.round(Nmetrics.acc(y, RC_q), 5)
                nmi1 = np.round(Nmetrics.nmi(y, RC_q), 5)
                vmea1 = np.round(Nmetrics.vmeasure(y, RC_q), 5)
                ari1 = np.round(Nmetrics.ari(y, RC_q), 5)
                print('rc', 'ACC=%.5f, NMI=%.5f, V-Measure=%.5f, ARI=%.5f' % (acc1, nmi1, vmea1, ari1))
                RC_Q = []
                RC_Q_rec = []
                RC_P = []
                RC_P_rec = []
                RC_q = self.rc.predict(input_dic)
                RC_p = self.rc.predict(input_dic).argmax(1)
                RC_Q.append(RC_q)
                for i in range(1449):
                    RC_p[i] = self.RC_target_distribution(RC_q[i])
                RC_P.append(RC_p)
                rec_data = self.autoencoder.predict(input_dic)
                RC_q_rec = self.rc.predict(rec_data)
                RC_p_rec = self.rc.predict(rec_data).argmax(1)
                RC_Q_rec.append(RC_q_rec)
                for i in range(1449):
                    RC_p_rec[i] = self.RC_target_distribution(RC_q_rec[i])
                # RC_p_rec = self.RC_target_distribution(RC_q_rec)
                RC_P_rec.append(RC_p_rec)
            # train on batch
            # RC_Q = []
            # RC_Q_rec = []
            # RC_P = []
            # RC_P_rec = []
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x[0].shape[0])]
            #print('s',idx,x[1][idx])
            x_batch = []
            y_batch = []
            x_batchrec = []
            y_batchrec = []

            for view in range(len(x)):
                x_batch.append(x[view][idx])
                y_batch.append(P[view][idx])
                y_batch.append(x[view][idx])
                y_batch.append(np.ones((len(idx), 1)))
            rec_data = self.autoencoder.predict(x_batch)

            # RC_q = self.rc.predict(x_batch)
            # RC_p = self.rc.predict(x_batch).argmax(1)
            # RC_Q.append(RC_q )
            # for i in range(len(x_batch[0])):
            #     RC_p[i] = self.RC_target_distribution(RC_q[i])
            #RC_p =self.RC_target_distribution(RC_q )
            # RC_P.append(RC_p)

            # RC_q_rec = self.rc.predict(rec_data)
            # RC_p_rec = self.rc.predict(rec_data).argmax(1)
            # RC_Q_rec.append(RC_q_rec)
            # for i in range(len(x_batch[0])):
            #     RC_p_rec[i] = self.RC_target_distribution(RC_q_rec[i])
            # # RC_p_rec = self.RC_target_distribution(RC_q_rec)
            # RC_P_rec.append(RC_p_rec)

            # y_batch.append(RC_P[0])
            # y_batch.append(RC_Q[0])
            y_batch.append(RC_p[idx])
            y_batch.append(RC_q[idx])
            valid = np.ones((len(idx), 1))
            fake = np.zeros((len(idx), 1))

            for v in range(len(x)):
                self.dis[v].train_on_batch(rec_data[v],fake)
                self.dis[v].train_on_batch(x_batch[v], valid)
            tmp = self.train_on_batch(xin=x_batch, yout=y_batch)
            if ite%10==0:
             ite_epoch.append(ite)
             all_loss.append(tmp[0])
            if ite%1000==0:

                sio.savemat('loss.mat', {'epoch': ite_epoch, 'loss': all_loss})
            #  镜像
            x_batch_rec = self.autoencoder.predict(x_batch)
            for view in range(len(x)):
                x_batchrec.append(x_batch_rec[view])
                y_batchrec.append(P_rec[view][idx])
                y_batchrec.append(x[view][idx])
                y_batchrec.append(np.ones((len(idx), 1)))
            y_batchrec.append(RC_p_rec[idx])
            y_batchrec.append(RC_q[idx])


            rec_rec_data = self.autoencoder.predict(x_batch_rec)
            for v in range(len(x)):
                self.dis[v].train_on_batch(rec_rec_data[v],fake)
                self.dis[v].train_on_batch(x_batch[v], valid)
            tmp1 = self.train_on_batch(xin=x_batchrec, yout=y_batchrec)  # [sum, q, xn, q, x]
            # print(tmp)

            KLLoss = []
            for view in range(len(x)):
                Loss[view] += tmp[3 * view + 2]       # lr
                kl_loss[view] += tmp[3 * view + 1]    # lc
                # KLLoss += tmp[2 * view + 1]
                KLLoss.append(tmp[3 * view + 1])
            MVKLLoss.append(KLLoss)
            index = index + 1 if (index + 1) * batch_size < x[0].shape[0] else 0
            #print('ss',index)
            #print(ite)
            ite += 1
            if ite >= int(maxiter):
                # break
                ite = 0
                print("self.autoencoder.load_weights(args.pretrain_dir)")
                self.autoencoder.load_weights(arg.pretrain_dir)
                features = self.encoder.predict(input_dic)
                for view in range(len(x)):
                    kmeans.fit_predict(features[view])
                    self.model.get_layer(name='clustering' + str(view + 1)).set_weights([kmeans.cluster_centers_])

        # save the trained model

        logfile.close()
        print('saving model to:', save_dir + '/model_final.h5')
        self.model.save_weights(save_dir + '/model_final.h5')
        print('Clustering time: %ds' % (time() - t1))
        print('End clustering:', '-' * 60)

        Q_and_X = self.model.predict(input_dic)
        y_pred = []
        for view in range(len(x)):
            y_pred.append(Q_and_X[view * 3].argmax(1))

        y_q = Q_and_X[(len(x) - 1) * 3]
        for view in range(len(x) - 1):
            y_q += Q_and_X[view * 3]
        # y_q = y_q/len(x)
        features = self.encoder.predict(input_dic)
        z = np.hstack(features)
        y_pred_mean = kmean1.fit_predict(z)
        return y_pred, y_pred_mean

    def new_P(self, inputs, centers):
        alpha = 1
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
        q **= (alpha + 1.0) / 2.0
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
        return q


