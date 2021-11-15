from Parameters import *
from keras import backend as K
from keras.layers.convolutional import *
from keras.layers import * 
from keras.layers.recurrent import *
from keras.layers.merge import *

def ctcLambdaFunction(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def modelForWords():
    img_w = word_cfg['img_w']
    img_h = word_cfg['img-h']
    max_text_len = word_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        inp_shape = (1, img_w, img_h)
    else:
        inp_shape = (img_w, img_h, 1)
    
    inp_data = Input(name = 'the_input', shape = inp_shape, dtype = 'float32')
    conv1 = Conv2D(64, strides = (3,3), padding = 'same', name = 'conv1', kernel_initializer= 'he_normal')(inp_data)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    mp1 = MaxPooling2D(pool_size = (2,2), name = 'max1')(act1)

    conv2 = Conv2D(128, strides = (3,3), padding = 'same', name = 'conv2', kernel_initializer= 'he_normal')(mp1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    mp2 = MaxPooling2D(pool_size = (2,2), name = 'max2')(act2)

    conv3 = Conv2D(256, strides = (3,3), padding = 'same', name = 'conv3', kernel_initializer= 'he_normal')(mp2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)
    conv4 = Conv2D(256, strides = (3,3), padding = 'same', name = 'conv4', kernel_initializer= 'he_normal')(act3)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)
    mp3 = MaxPooling2D(pool_size = (1,2), name = 'max3')(act4)

    conv5 = Conv2D(512, strides = (3,3), padding = 'same', name = 'conv5', kernel_initializer= 'he_normal')(mp3)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)
    conv6 = Conv2D(512, strides = (3,3), padding = 'same', name = 'conv6', kernel_initializer= 'he_normal')(act5)
    bn6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(bn6)
    mp4 = MaxPooling2D(pool_size = (1,2), name = 'max4')(act6)

    conv7 = Conv2D(512, (2, 2), padding='same', name='conv7', kernel_initializer='he_normal')(mp4)
    bn7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(bn7)
