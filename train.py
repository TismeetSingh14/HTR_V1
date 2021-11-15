from Parameters import *
from DataGenerator import DataGenerator
from Model import modelForWords
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from Utils import *

def train(train_data, val_data, is_word_model):
    if is_word_model:
        model, _ = modelForWords()
        cfg = word_cfg
    
    input_length = cfg['input_length']
    model_name = cfg['model_name']
    max_text_len = cfg['max_text_len']
    img_w = cfg['img_w']
    img_h = cfg['img_h']
    batch_size = cfg['batch_size']
    train_set = DataGenerator(train_data, img_h, img_w, batch_size, input_length, max_text_len)
    print('Loading Data')
    train_set.generate()

    val_set = DataGenerator(val_data, img_h, img_w, batch_size, input_length, max_text_len)
    print('Loading Data')
    val_set.generate()

    print("Number train samples: ", train_set.n)
    print("Number val samples: ", val_set.n)

    model.compile(loss = {'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    checkPoint = ModelCheckpoint(
        filepath = 'Resource/' + model_name + '--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
        verbose = 1, save_best_only=True, save_weights_only=True
    )

    earlyStop = EarlyStopping(
        monitor = 'val_loss', min_delta=0, patience=10, verbose=0, mode = 'min'
    )

    model.fit_generator(generator = train_set.nextBatch(),
                        steps_per_epoch = train_set.n // batch_size,
                        epochs = 32,
                        validation_data=val_set.nextBatch(),
                        validation_steps=val_set.n // batch_size,
                        callbacks = [checkPoint, earlyStop]) 
    
    return model

train_data = getPathAndTexts('data/IAM/splits/train.uttlist', is_words=True)
val_data = getPathAndTexts('data/IAM/splits/validation.uttlist', is_words=True)
print('number of train image: ', len(train_data))
print('number of valid image: ', len(val_data))

model = train(train_data, val_data, True)