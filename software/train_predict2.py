import pandas as pd
import os
import tensorflow as tf
import pickle
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)]
)
# tf.compat.v1.disable_eager_execution()
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config=tf.compat.v1.ConfigProto()
#
# config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
#
# sess = tf.compat.v1.Session(config=config)

# tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import *
from tensorflow.keras import layers
# from keras import losses
# from keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage import morphology
from shutil import copyfile
from skimage import io
from data3 import *
import warnings
import skimage
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
# from scipy import ndimage
# from keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import Callback
# from sklearn.metrics import accuracy_score,f1_score


act = 'relu'
dropout_rate = 0.5


def iou(y_true, y_pred, smooth=1e-5):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_true=y_true > 0.5
    y_pred=y_pred > 0.5
    y_true=y_true.astype('float32')
    y_pred=y_pred.astype('float32')
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def dice_coef(y_true, y_pred,smooth=1):
    y_truef = K.flatten(y_true)  # 将y_true拉为一维
    y_predf = K.flatten(y_pred)
    intersection = K.sum(y_truef * y_predf)
    return (2 * intersection+smooth) / (K.sum(y_truef) + K.sum(y_predf)+smooth)
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_true = y_true > 0.5
        y_pred = y_pred > 0.5
        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        y_true = y_true > 0.5
        y_pred = y_pred > 0.5
        y_true = y_true.astype('float32')
        y_pred = y_pred.astype('float32')
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def sensitivity(y_true, y_pred):
    beta=1
    smooth = 1e-7
    y_pred = K.clip(y_pred, 0, 1)



    tp = K.sum(y_true * y_pred, axis=[0,1,2])
    fp = K.sum(y_pred, axis=[0,1,2]) - tp
    fn = K.sum(y_true, axis=[0,1,2]) - tp

    score = ((1 + beta ** 2) * tp) / (tp + fn + smooth)
    # score = tf.reduce_mean(score)
    sensitivity_coef=score

    return sensitivity_coef
def getBinaryTensor(imgTensor, boundary = 0.5):
    one = tf.ones_like(imgTensor)
    zero = tf.zeros_like(imgTensor)
    return tf.where(imgTensor > boundary, one, zero)
def specify(y_true, y_pred):
    smooth=1e-7
    y_pred=getBinaryTensor(y_pred)
    allTrue = K.sum(K.clip(y_true + y_pred, 0, 1))

    score = 2 * (512 * 512 - allTrue) / (512 * 512 - K.sum(y_true) + smooth)

    return score


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_true=y_true > 0.5
    y_pred=y_pred > 0.5
    y_true=y_true.astype('float32')
    y_pred=y_pred.astype('float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def get_contours(img):
    img=img.squeeze()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

def HD95(y_true,y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    # y_true=y_true > 0.5
    # y_pred=y_pred > 0.5
    y_true=y_true.astype('float32')
    y_pred=y_pred.astype('float32')
    y_true=(y_true*255).astype(np.uint8)
    y_pred =(y_pred*255).astype(np.uint8)
    cnt_cs1 = get_contours(y_true)
    cnt_cs2 = get_contours(y_pred)
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    d1 = hausdorff_sd.computeDistance(cnt_cs1,cnt_cs2)
    return d1
from PyQt5.QtCore import *
from tensorflow.keras.callbacks import Callback
class LossHistory(QThread, Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
        iters = range(len(self.losses['epoch']))
        plt.figure()
        # acc
        plt.xlim(0, 49)
        plt.plot(iters, self.accuracy['epoch'], 'r', label='train acc')
        plt.plot(iters, self.val_acc['epoch'], 'b', label='val acc')
        # loss
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        plt.title('Model Accuracy')
        plt.savefig('Figure_123.png')
        plt.figure()
        plt.xlim(0, 49)
        plt.plot(iters, self.losses['epoch'], 'r', label='train loss')
        plt.plot(iters, self.val_loss['epoch'], 'b', label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Loss')
        plt.legend(loc="upper right")
        plt.savefig('Figure_1234.png')
        from PyQt5 import QtWidgets
        QtWidgets.QApplication.processEvents()
    # def run(self):
    #     from PyQt5 import QtWidgets, QtCore, QtGui, Qt
    #     from PyQt5.QtWidgets import *
    #     from PyQt5.QtGui import *
    #     from PyQt5.QtCore import *
    #     self.graph1 = QtWidgets.QLabel(self.frame)
    #     self.graph1.setObjectName("label_10")
    #     self.gridLayout_3.addWidget(self.graph1, 0, 0, 1, 1)
    #     self.graph2 = QtWidgets.QLabel(self.frame)
    #     self.graph2.setObjectName("label_11")
    #     self.gridLayout_3.addWidget(self.graph2, 0, 1, 1, 1)
    #     self.gridLayout.addWidget(self.frame, 0, 1, 1, 1)
    #     pix = QPixmap("Figure_123.png")
    #     self.graph1.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
    #     self.graph1.setAlignment(Qt.AlignCenter)
    #     self.graph1.setScaledContents(True)  # 设置图片自适应窗口大小
    #
    #     pix = QPixmap("Figure_1234.png")
    #     self.graph2.setPixmap(pix.scaled(640, 480, QtCore.Qt.KeepAspectRatio))
    #     self.graph2.setAlignment(Qt.AlignCenter)
    #     self.graph2.setScaledContents(True)  # 设置图片自适应窗口大小

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
class NestNet(QThread):
    signal_res = pyqtSignal(dict)
    def __init__(self,img_rows,img_cols,color_type,num_class,deep_supervision, method, datafile, modelname, augment, models, epoch, lr, optimizers):
        super(NestNet, self).__init__()
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.color_type = color_type
        self.num_class = num_class
        self.deep_supervision = deep_supervision
        self.datafile = datafile
        self.method = method
        self.modelname = modelname
        self.augment = augment
        self.models = models
        self.epoch = epoch
        self.lr = lr
        self.optimizers = optimizers

    @property
    def get_NestNet(self):
        def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

            x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1',
                       kernel_initializer='he_normal', padding='same')(input_tensor)
            # x = layers.BatchNormalization()(x)
            # x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
            x1 = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_2',
                        kernel_initializer='he_normal', padding='same')(x)
            # x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)
            # x1 = layers.BatchNormalization()(x1)

            return x1

        nb_filter = [32, 64, 128, 256, 512]

        # Handle Dimension Ordering for different backends
        '''
        global bn_axis
        if K.image_dim_ordering() == 'channels_last':
          bn_axis = 3
          img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
        else:
          bn_axis = 1
          img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')'''
        img_input = Input(shape=(self.img_rows, self.img_cols, self.color_type))
        conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])  # 512 512 32
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)  # 256 256 32

        conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])  # 256 256 64
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)  # 128 128 64

        up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
        conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

        conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

        up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
        conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

        up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
        conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

        conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

        up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
        conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

        up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
        conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

        up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
        conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

        conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

        up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
        conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

        up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
        conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

        up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
        conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

        up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
        conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

        nestnet_output_1 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_1',
                                  kernel_initializer='he_normal', padding='same')(conv1_2)
        nestnet_output_2 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_2',
                                  kernel_initializer='he_normal', padding='same')(conv1_3)
        nestnet_output_3 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_3',
                                  kernel_initializer='he_normal', padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_4',
                                  kernel_initializer='he_normal', padding='same')(conv1_5)

        if self.deep_supervision:
            model = Model(inputs=img_input, outputs=[nestnet_output_1,
                                                     nestnet_output_2,
                                                     nestnet_output_3,
                                                     nestnet_output_4])
        else:
            model = Model(inputs=img_input, outputs=[nestnet_output_3])

        return model
    def deeplabV3(self):
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def relu6(x):
            return relu(x, max_value=6)

        def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
            in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
            x = inputs
            prefix = 'expanded_conv_{}_'.format(block_id)
            if block_id:
                # Expand

                x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                           use_bias=False, activation=None,
                           name=prefix + 'expand')(x)
                x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                       name=prefix + 'expand_BN')(x)
                x = Activation(relu6, name=prefix + 'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'
            # Depthwise
            x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                                use_bias=False, padding='same', dilation_rate=(rate, rate),
                                name=prefix + 'depthwise')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'depthwise_BN')(x)

            x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

            # Project
            x = Conv2D(pointwise_filters,
                       kernel_size=1, padding='same', use_bias=False, activation=None,
                       name=prefix + 'project')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'project_BN')(x)

            if skip_connection:
                return Add(name=prefix + 'add')([inputs, x])

            # if in_channels == pointwise_filters and stride == 1:
            #    return Add(name='res_connect_' + str(block_id))([inputs, x])

            return x
    def unet3plus(self):
        def aggregate(l1, l2, l3, l4, l5):
            out = Concatenate(axis=-1)([l1, l2, l3, l4, l5])
            out = Convolution2D(320, 3, activation='relu', padding='same', kernel_initializer='he_normal')(out)
            out = BatchNormalization()(out)
            # out = ReLU(out)  # 这个的原因导致 ：NoneType' object has no attribute '_inbound_nodes；
            # out = add(ReLU(out))

            return out

        conv_num = 32
        inputs = Input((512, 512, 2))
        XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(XE1)
        XE1_pool = MaxPooling2D(pool_size=(2, 2))(XE1)

        XE2 = Convolution2D(conv_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE1_pool)
        XE2 = Convolution2D(conv_num * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(XE2)
        XE2_pool = MaxPooling2D(pool_size=(2, 2))(XE2)

        XE3 = Convolution2D(conv_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE2_pool)
        XE3 = Convolution2D(conv_num * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(XE3)
        XE3_pool = MaxPooling2D(pool_size=(2, 2))(XE3)

        XE4 = Convolution2D(conv_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE3_pool)
        XE4 = Convolution2D(conv_num * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(XE4)
        XE4 = Dropout(0.5)(XE4)
        XE4_pool = MaxPooling2D(pool_size=(2, 2))(XE4)

        XE5 = Convolution2D(conv_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE4_pool)
        XE5 = Convolution2D(conv_num * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(XE5)
        XE5 = Dropout(0.5)(XE5)

        XD4_from_XE5 = UpSampling2D(size=(2, 2), interpolation='bilinear')(XE5)
        XD4_from_XE5 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD4_from_XE5)
        XD4_from_XE4 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE4)
        XD4_from_XE3 = MaxPooling2D(pool_size=(2, 2))(XE3)
        XD4_from_XE3 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD4_from_XE3)
        XD4_from_XE2 = MaxPooling2D(pool_size=(4, 4))(XE2)
        XD4_from_XE2 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD4_from_XE2)
        XD4_from_XE1 = MaxPooling2D(pool_size=(8, 8))(XE1)
        XD4_from_XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD4_from_XE1)
        XD4 = aggregate(XD4_from_XE5, XD4_from_XE4, XD4_from_XE3, XD4_from_XE2, XD4_from_XE1)

        XD3_from_XE5 = UpSampling2D(size=(4, 4), interpolation='bilinear')(XE5)
        XD3_from_XE5 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD3_from_XE5)
        XD3_from_XD4 = UpSampling2D(size=(2, 2), interpolation='bilinear')(XD4)
        XD3_from_XD4 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD3_from_XD4)
        XD3_from_XE3 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE3)
        XD3_from_XE2 = MaxPooling2D(pool_size=(2, 2))(XE2)
        XD3_from_XE2 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD3_from_XE2)
        XD3_from_XE1 = MaxPooling2D(pool_size=(4, 4))(XE1)
        XD3_from_XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD3_from_XE1)
        XD3 = aggregate(XD3_from_XE5, XD3_from_XD4, XD3_from_XE3, XD3_from_XE2, XD3_from_XE1)

        XD2_from_XE5 = UpSampling2D(size=(8, 8), interpolation='bilinear')(XE5)
        XD2_from_XE5 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD2_from_XE5)
        XD2_from_XE4 = UpSampling2D(size=(4, 4), interpolation='bilinear')(XE4)
        XD2_from_XE4 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD2_from_XE4)
        XD2_from_XD3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(XD3)
        XD2_from_XD3 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD2_from_XD3)
        XD2_from_XE2 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XE2)
        XD2_from_XE1 = MaxPooling2D(pool_size=(2, 2))(XE1)
        XD2_from_XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            XD2_from_XE1)
        XD2 = aggregate(XD2_from_XE5, XD2_from_XE4, XD2_from_XD3, XD2_from_XE2, XD2_from_XE1)
        XD2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(XD2)

        # XD1_from_XE5 = UpSampling2D(size=(16, 16), interpolation='bilinear')(XE5)
        # XD1_from_XE5 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     XD1_from_XE5)
        # XD1_from_XE4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(XE4)
        # XD1_from_XE4 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     XD1_from_XE4)
        # XD1_from_XE3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(XE3)
        # XD1_from_XE3 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     XD1_from_XE3)
        # XD1_from_XD2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(XD2)
        # XD1_from_XD2 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     XD1_from_XD2)
        # XD1_from_XE1 = Convolution2D(conv_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        #     XE1)
        # XD1 = aggregate(XD1_from_XE5, XD1_from_XE4, XD1_from_XE3, XD1_from_XD2, XD1_from_XE1)

        # out = Convolution2D(conv_num * 5, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(XD1)
        out = Convolution2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(XD2)
        model = Model(inputs=inputs, outputs=out)

        return model

    def get_unet(self):
        inputs = layers.Input(shape=(self.img_rows, self.img_cols, 2))  # 通道个数
        conv1 = Convolution2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = Convolution2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = Convolution2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 256*256*64

        conv3 = Convolution2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = Convolution2D(128, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 128*128*128

        conv4 = Convolution2D(256, 3, activation='relu', padding='same')(pool3)
        conv4 = Convolution2D(256, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 64*64*256

        conv5 = Convolution2D(512, 3, activation='relu', padding='same')(pool4)
        conv5 = Convolution2D(512, 3, activation='relu', padding='same')(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)  # 32*32*512

        convdeep = Convolution2D(1024, 3, activation='relu', padding='same')(pool5)
        convdeep = Convolution2D(1024, 3, activation='relu', padding='same')(convdeep)  # 32*32*1024

        upmid = concatenate([Convolution2D(512, 2, padding='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5],
                            axis=3)  # 64*64*1024
        convmid = Convolution2D(512, 3, activation='relu', padding='same')(upmid)
        convmid = Convolution2D(512, 3, activation='relu', padding='same')(convmid)  # 64*64*512

        up6 = concatenate(
            [Convolution2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(convmid)), conv4],
            axis=3)  # 128*128*512
        conv6 = Convolution2D(256, 3, activation='relu', padding='same')(up6)
        conv6 = Convolution2D(256, 3, activation='relu', padding='same')(conv6)  # 128*128*256

        up7 = concatenate(
            [Convolution2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3],
            axis=3)  # 256*256*256
        conv7 = Convolution2D(128, 3, activation='relu', padding='same')(up7)
        conv7 = Convolution2D(128, 3, activation='relu', padding='same')(conv7)  # 256*256*128

        up8 = concatenate(
            [Convolution2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2],
            axis=3)  # 512*512*128
        conv8 = Convolution2D(64, 3, activation='relu', padding='same')(up8)
        conv8 = Convolution2D(64, 3, activation='relu', padding='same')(conv8)  # 512*512*64

        up9 = concatenate(
            [Convolution2D(32, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1],
            axis=3)
        conv9 = Convolution2D(32, 3, activation='relu', padding='same')(up9)
        conv9 = Convolution2D(32, 3, activation='relu', padding='same')(conv9)

        conv10 = Convolution2D(1, 1, activation='sigmoid')(conv9)  # 512*512*1

        model = Model(inputs=inputs, outputs=conv10)
        return model


    def run(self):

        print('predict test data')
        # Modelfiles = os.listdir('./logs0')  # 列出路径下的文件
        if self.models == "U-net":
            model = self.get_unet()
        if self.models == "U-net++":
            model = self.get_NestNet()
        if self.models == "U-net+++":
            model = self.unet3plus()
        if self.models == "deeplab v3":
            model = self.deeplabV3()
        # model = self.get_unet_batchnormalization()
        if self.modelname == '':
            pass
        else:
            model.load_weights(self.modelname)

        model.summary()


        '''训练层冻结？'''
        if self.optimizers == 'Adam':
            model.compile(optimizer=optimizers.Adam(lr=float(self.lr)), loss='binary_crossentropy',run_eagerly=True,
                        metrics=['accuracy', f1, iou])
        if self.optimizers == 'SGD':
            model.compile(optimizer=optimizers.SGD(lr=float(self.lr)), loss='binary_crossentropy',run_eagerly=True,
                        metrics=['accuracy', f1, iou])
        if self.optimizers == 'RMSprop':
            model.compile(optimizer=optimizers.RMSprop(lr=float(self.lr)), loss='binary_crossentropy',run_eagerly=True,
                        metrics=['accuracy', f1, iou])
        if self.optimizers == 'Adagrad':
            model.compile(optimizer=optimizers.Adagrad(lr=float(self.lr)), loss='binary_crossentropy',run_eagerly=True,
                        metrics=['accuracy', f1, iou])
        if self.optimizers == 'Adamax':
            model.compile(optimizer=optimizers.Adamax(lr=float(self.lr)), loss='binary_crossentropy',run_eagerly=True,
                        metrics=['accuracy', f1, iou])
        # class TestHistory(Callback):
        #     def on_epoch_end(self,epoch,epoch_logs):
        #         model_path = './logs1/'
        #         modelfiles = os.listdir(model_path)
        #         model.load_weights(model_path + modelfiles[epoch-1])
        #         model.compile(optimizer=optimizers.Adam(lr=0.00005), loss='binary_crossentropy',
        #                       metrics=['accuracy', dice_coef, f1])
        #         print('predict test data')
        #         test_dataset = './test dataA'
        #         test_dataset_data = './test dataA/data'
        #         test_dataset_data1 = './test dataA/data1'
        #         test_dataset_label = './test dataA/label'
        #         testfiles = os.listdir(test_dataset_data)
        #         test_dataset = './test dataA'
        #         test_dataset_data = './test dataA/data'
        #         testfiles = os.listdir(test_dataset_data)
        #         score = np.zeros(len(testfiles))
        #         score1 = np.zeros(len(testfiles))
        #         score2 = np.zeros(len(testfiles))
        #         score3 = np.zeros(len(testfiles))
        #         all_score=np.zeros([50,4])
        #         for i in range(len(testfiles)):
        #             # data preprocess
        #             image = skimage.io.imread(os.path.join(test_dataset_data, str(i + 1).zfill(4) + '.tif'))
        #             image1 = skimage.io.imread(os.path.join(test_dataset_data1, str(i + 1).zfill(4) + '.tif'))
        #             test1 = skimage.io.imread(os.path.join(test_dataset_label, str(i + 1).zfill(4) + '.tif'))
        #             # image = skimage.io.imread(os.path.join(dataset_data, testfiles[i]))
        #             # image1 = skimage.io.imread(os.path.join(dataset_data1, testfiles[i]))
        #             test1 = cv2.resize(test1, (512, 512))
        #             test1 = test1 > 0
        #             test1 = test1.astype('float32')
        #             images = np.zeros((image.shape[0], image.shape[1], 2))
        #             images[:, :, 0] = image
        #             images[:, :, 1] = image1
        #             image = images
        #             image = cv2.resize(image, (512, 512))
        #             image = image.astype('float32')
        #             mean = np.mean(image)
        #             std = np.std(image)
        #             image -= mean
        #             image /= std
        #             test = np.zeros(shape=(1, 512, 512, 2), dtype='float32')
        #             test2 = np.zeros(shape=(1, 512, 512), dtype='float32')
        #             test2[0, :, :] = test1
        #             test[0, :, :, :] = image
        #             test=model.predict(test)
        #             f1score=f1(test2,test)
        #             with tf.compat.v1.Session() as sess:
        #                 score[i]=f1score.eval()
        #         f1_score=np.mean(score)
        #         all_score[epoch-1,0]=f1_score
        #         if all_score[0, 0] != 0:
        #             df = pd.DataFrame(all_score)
        #             df.to_csv('./result/epoch.csv')





        print('Fitting model...')

        '''data_path ='.\Data1\W-1000\A\SE'
        data_path1 ='./DataB\\biaozhu\\adjustGP\\HD'
        label_path = './DataB\\biaozhu\\adjustGP\\label'''
        data_path ='.\Data1\W-1000\A\SE'
        data_path1 ='.\Data1\W-1000\A\HD'
        label_path = '.\Data1\W-1000\A\label'

        # data_path = 'data'
        # data_path1 = 'data2'
        # label_path = 'label'
        # if not os.path.exists('data'):
        #     os.makedirs('data')
        #     os.makedirs('data2')
        #     os.makedirs('label')
        # files = os.listdir(data_path) # 列出路径下的文件
        # np.random.shuffle(files)
        # for i in range(len(files)):
        #     copyfile(os.path.join(data_path,files[i]),
        #              os.path.join('data',str(i+1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join(data_path1, files[i]),
        #              os.path.join('data2', str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join(label_path, files[i]),
        #              os.path.join('label', str(i + 1).rjust(4, '0') + '.tif'))
        # data_path = 'data'
        # data_path1 = 'data2'
        # label_path = 'label'
        # if not os.path.exists(data_path):
        #     os.makedirs(data_path)
        #     os.makedirs(data_path1)
        #     os.makedirs(label_path)
        #
        # files = os.listdir(data_path)
        # get_key = lambda i: int(i.split('.')[0])
        # files = sorted(files, key=get_key)
        # files_number = len(files)
        #
        #
        train_dataset = self.datafile+'/train dataA'
        print(train_dataset)
        val_dataset = self.datafile+'/val dataA' # 验证集路径
        test_dataset = self.datafile+'/test dataA'  # 验证集路径
        # if not os.path.exists(train_dataset):
        #     os.makedirs(train_dataset)
        #     os.makedirs(val_dataset)
        #     os.makedirs(test_dataset)
        #
        #
        train_dataset_data = train_dataset+'\data'
        train_dataset_data1 = train_dataset+'\data1'
        val_dataset_data = val_dataset+'\data'
        val_dataset_data1 = val_dataset+'\data1'
        test_dataset_data = './test dataA\\data'
        test_dataset_data1 = './test dataA\\data1'
        train_dataset_label = train_dataset+'\label'
        val_dataset_label = val_dataset+'\label'
        test_dataset_label = './test dataA\\label'
        if not os.path.exists(train_dataset_data):
            os.makedirs(train_dataset_data)
            os.makedirs(val_dataset_data)
            os.makedirs(train_dataset_data1)
            os.makedirs(val_dataset_data1)
            os.makedirs(train_dataset_label)
            os.makedirs(val_dataset_label)
            os.makedirs(test_dataset_data)
            os.makedirs(test_dataset_data1)
            os.makedirs(test_dataset_label)
        # files = os.listdir('dataA/data')
        # np.random.shuffle(files)
        # for i in range(len(files)):
        #     copyfile(os.path.join('dataA/data', files[i]),
        #                  os.path.join(test_dataset_data, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('dataA/data1', files[i]),
        #                  os.path.join(test_dataset_data1, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('dataA/label', files[i]),
        #                  os.path.join(test_dataset_label, str(i + 1).rjust(4, '0') + '.tif'))
        #
        # if not os.path.exists(train_dataset_data):
        #     os.makedirs(train_dataset_data)
        #     os.makedirs(val_dataset_data)
        #     os.makedirs(train_dataset_data1)
        #     os.makedirs(val_dataset_data1)
        #     os.makedirs(train_dataset_label)
        #     os.makedirs(val_dataset_label)
        #     os.makedirs(test_dataset_data)
        #     os.makedirs(test_dataset_data1)
        #     os.makedirs(test_dataset_label)
        #     Cnt = 0
        #     Cnt1 = 0
        #     Cnt2 = 0
        # trainfiles = os.listdir('./data')
        # np.random.shuffle(trainfiles)
        # for i in range(8):
        #     copyfile(os.path.join('./data', trainfiles[i]),
        #                  os.path.join(train_dataset_data, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('./data2', trainfiles[i]),
        #                  os.path.join(train_dataset_data1, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('./label', trainfiles[i]),
        #                  os.path.join(train_dataset_label, str(i + 1).rjust(4, '0') + '.tif'))
        # for i in range(10):
        #     copyfile(os.path.join('./data', trainfiles[i+8]),
        #                      os.path.join(val_dataset_data, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('./data2', trainfiles[i+8]),
        #                      os.path.join(val_dataset_data1, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('./label', trainfiles[i+8]),
        #                      os.path.join(val_dataset_label, str(i + 1).rjust(4, '0') + '.tif'))
        # for i in range(int(len(trainfiles)*0.3)):
        # #     copyfile(os.path.join('data', trainfiles[i+int(len(trainfiles)*0.6)]),
        # #                  os.path.join(val_dataset_data, str(i + 1).rjust(4, '0') + '.tif'))
        # #     copyfile(os.path.join('data2', trainfiles[i+int(len(trainfiles)*0.6)]),
        # #                  os.path.join(val_dataset_data1, str(i + 1).rjust(4, '0') + '.tif'))
        # #     copyfile(os.path.join('label', trainfiles[i+int(len(trainfiles)*0.6)]),
        # #                  os.path.join(val_dataset_label, str(i + 1).rjust(4, '0') + '.tif'))
        # for i in range(len(trainfiles)-28):
        #     copyfile(os.path.join('data', trainfiles[i+28]),
        #                  os.path.join(test_dataset_data, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('data2', trainfiles[i+28]),
        #                  os.path.join(test_dataset_data1, str(i + 1).rjust(4, '0') + '.tif'))
        #     copyfile(os.path.join('label', trainfiles[i+28]),
        #                  os.path.join(test_dataset_label, str(i + 1).rjust(4, '0') + '.tif'))

            # for i in range(files_number):
            #         random_value = np.random.random()
            #         if random_value < 0.2:
            #             Cnt += 1
            #             copyfile(os.path.join(data_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(val_dataset_data, str(Cnt).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(data_path1, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(val_dataset_data1, str(Cnt).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(label_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(val_dataset_label, str(Cnt).rjust(4, '0') + '.tif'))
            #         elif random_value < 0.8:
            #             Cnt1 += 1
            #             copyfile(os.path.join(data_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(train_dataset_data, str(Cnt1).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(data_path1, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(train_dataset_data1, str(Cnt1).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(label_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(train_dataset_label, str(Cnt1).rjust(4, '0') + '.tif'))
            #         else:
            #             Cnt2 += 1
            #             copyfile(os.path.join(data_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(test_dataset_data, str(Cnt2).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(data_path1, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(test_dataset_data1, str(Cnt2).rjust(4, '0') + '.tif'))
            #             copyfile(os.path.join(label_path, str(i + 1).rjust(4, '0') + '.tif'),
            #                      os.path.join(test_dataset_label, str(Cnt2).rjust(4, '0') + '.tif'))
        trainfiles = os.listdir(train_dataset_data)# 列出路径下的文件
        valfiles = os.listdir(val_dataset_data)

        train_generator = data_generator(train_dataset, len(trainfiles), shuffle=True, augment=self.augment, batch_size=2)
        val_generator = data_generator(val_dataset, len(valfiles), shuffle=True, batch_size=2, augment=True)

        model_checkpoint = ModelCheckpoint(
            './logs1\\nestnetGP200bcadam_{epoch:04d}_{loss:.05f}_{val_loss:.05f}_{iou:.05f}_{val_iou:.05f}_{accuracy:.05f}_{val_accuracy:.05f}.h5',
            monitor='val_accuracy', verbose=1, save_best_only=False)  # 保存权重路径
        model_history = LossHistory()
        if not os.path.exists('./logs1/'):
            os.makedirs('./logs1/')

        print(1)
        #Tensorboard = TensorBoard(log_dir="model")
        #steps_per_epoch=2000代表每个epoch要执行生成器2000次，并且如果生成器的batch_size=128，那么执行2000次生成器中每次生成128的小批量输入网络中训练
        history = model.fit_generator(train_generator,steps_per_epoch=100,validation_data=val_generator,validation_steps=100,
                            callbacks=[model_checkpoint, model_history],epochs=int(self.epoch), verbose=1
                            )
        with open('trainHistoryDict.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        # plt.figure()
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # # plt.plot(history.history['f1'])
        # # plt.plot(history.history['val_f1'])
        # # plt.plot(history.history['loss'])
        # # plt.plot(history.history['val_loss'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train loss', 'Val loss'], loc='upper left')
        # plt.show()
        # # plt.plot(history.history['accuracy'])
        # # plt.plot(history.history['val_accuracy'])
        # # plt.plot(history.history['f1'])
        # # plt.plot(history.history['val_f1'])
        # plt.figure()
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train loss', 'Val loss'], loc='upper left')
        # plt.show()

        #model.save_weights('mymodel.h5')

        # print('predict test data')
        # Modelfiles = os.listdir(r'E:/测试/W-1000 3 RL')  # 列出路径下的文件
        # ModelName = Modelfiles[len(Modelfiles) - 3]
        # model.load_weights(r'E:/测试/W-1000 3 RL/' + ModelName)
        # dataset_data = r'E:\测试\test dataA\data'
        # dataset_data1 = r'E:\测试\test dataA\data1'
        # testfiles = os.listdir(dataset_data)
        # path = './result'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # for i in range(len(testfiles)):
        #     # data preprocess
        #     image = skimage.io.imread(os.path.join(dataset_data, str(i+1).zfill(4) + '.tif'))
        #     image1 = skimage.io.imread(os.path.join(dataset_data1, str(i+1).zfill(4) + '.tif'))
        #     #image = skimage.io.imread(os.path.join(dataset_data, testfiles[i]))
        #     #image1 = skimage.io.imread(os.path.join(dataset_data1, testfiles[i]))
        #     images = np.zeros((image.shape[0], image.shape[1], 2))
        #     images[:, :, 0] = image
        #     images[:, :, 1] = image1
        #     image = images
        #     image = cv2.resize(image, (512, 512))
        #     image = image.astype('float32')
        #     mean = np.mean(image)
        #     std = np.std(image)
        #     image -= mean
        #     image /= std
        #     test = np.zeros(shape=(1, 512, 512, 2),dtype='float32')
        #     test[0, :, :, :] = image
        #     re = model.predict(test, batch_size=1, verbose=1)
        #     re = re.reshape(1, 512,512, 1)
        #     mask = re[0, :, :, 0]
        #     mask = mask > 0.5
        #     io.imsave('./result/'+str(i+1).rjust(4,'0')+'.tif', (mask*255).astype('uint8'))
            # io.imsave('./result1/' + testfiles[i], (re[0, :, :, 0] * 255).astype('uint8'))




if __name__ == '__main__':
    img_rows = 512
    img_cols = 512
    color_type = 2
    num_class = 2
    deep_supervision = False
    datafile = ''
    modelname = ''
    predict = NestNet(img_rows,img_cols,color_type,num_class,deep_supervision, datafile, modelname)
    predict.run()


