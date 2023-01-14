import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
def self_bce(y_true, y_pred):  # y_true p[idx] 1 / y_pred 1
    #print('a',y_true,y_pred)
    #Tensor("ExpandDims:0", shape=(1, 1), dtype=float32)
    #Tensor("model_5/rcclustering/Softmax:0", shape=(1, 5), dtype=float32)
    # 第九行 报错
    y_true=tf.cast(y_true,dtype=tf.float32)
    r = tf.where(y_true == 1., 1., -1.)
    lng = K.binary_crossentropy(y_true, K.max(y_pred, 1), from_logits=False)
    loss = -1e-4 * r * 0.5 * lng
    #K.mean(loss, axis=-1)
    return loss

