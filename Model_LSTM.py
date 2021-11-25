from Parameters import *
from keras import backend as K
from keras.layers.convolutional import *
from keras.layers import * 
from keras.layers.recurrent import *
from keras.layers.merge import *
from keras.models import *

def ctcLambdaFunction(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def modelForWords():
    img_w = word_cfg['img_w']
    img_h = word_cfg['img_h']
    max_text_len = word_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        inp_shape = (1, img_w, img_h)
    else:
        inp_shape = (img_w, img_h, 1)
    
    inp_data = Input(name = 'the_input', shape = inp_shape, dtype = 'float32')
    conv1 = Conv2D(64, kernel_size = (3,3), padding = 'same', name = 'conv1', kernel_initializer= 'he_normal')(inp_data)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    mp1 = MaxPooling2D(pool_size = (2,2), name = 'max1')(act1)

    conv2 = Conv2D(128, kernel_size = (3,3), padding = 'same', name = 'conv2', kernel_initializer= 'he_normal')(mp1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    mp2 = MaxPooling2D(pool_size = (2,2), name = 'max2')(act2)

    conv3 = Conv2D(256, kernel_size = (3,3), padding = 'same', name = 'conv3', kernel_initializer= 'he_normal')(mp2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)
    conv4 = Conv2D(256, kernel_size = (3,3), padding = 'same', name = 'conv4', kernel_initializer= 'he_normal')(act3)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)
    mp3 = MaxPooling2D(pool_size = (1,2), name = 'max3')(act4)

    conv5 = Conv2D(512, kernel_size = (3,3), padding = 'same', name = 'conv5', kernel_initializer= 'he_normal')(mp3)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)
    conv6 = Conv2D(512, kernel_size = (3,3), padding = 'same', name = 'conv6', kernel_initializer= 'he_normal')(act5)
    bn6 = BatchNormalization()(conv6)
    act6 = Activation('relu')(bn6)
    mp4 = MaxPooling2D(pool_size = (1,2), name = 'max4')(act6)

    conv7 = Conv2D(512, (2, 2), padding='same', name='conv7', kernel_initializer='he_normal')(mp4)
    bn7 = BatchNormalization()(conv7)
    act7 = Activation('relu')(bn7)

    rs1 = Reshape(target_shape=((32,2048)), name='reshape1')(act7)
    dc1 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(rs1)

    lstm1 = LSTM(128,return_sequences=True, kernel_initializer='he_normal', name='lstm1')(dc1)
    lstm1_ = LSTM(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_')(dc1)
    reversed_lstm1_ = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm1_)

    lstm1_merged = add([lstm1,reversed_lstm1_])
    lstm1_merged_1 = BatchNormalization()(lstm1_merged)

    lstm2 = LSTM(256,return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged_1)
    lstm2_ = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_')(lstm1_merged_1)
    reversed_lstm2_ = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm2_)
    
    lstm2_merged = concatenate([lstm2,reversed_lstm2_])
    lstm2_merged_1 = BatchNormalization()(lstm2_merged)

    lstm3 = LSTM(256,return_sequences=True, kernel_initializer='he_normal', name='lstm3')(lstm2_merged_1)
    lstm3_ = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm3_')(lstm2_merged_1)
    reversed_lstm3_ = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm3_)
    
    lstm3_merged = concatenate([lstm3,reversed_lstm3_])
    lstm3_merged_1 = BatchNormalization()(lstm3_merged)

    dc2 = Dense(classes, kernel_initializer='he_normal', name='dense2')(lstm3_merged_1)
    y_pred = Activation('softmax', name='softmax')(dc2)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctcLambdaFunction, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs = [inp_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=inp_data, outputs=y_pred)
    model_predict.summary()

    return model, model_predict