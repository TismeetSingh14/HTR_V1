import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import random
from keras import backend as K
from Preprocessor import preprocess

char_list = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
classes = len(char_list) + 1
word_cfg = {'batch_size': 64, 'input_length': 30, 'model_name': 'iam_words', 'max_text_len': 16, 'img_w': 128, 'img_h': 64}
line_cfg = {'batch_size': 16, 'input_length': 98, 'model_name': 'iam_line', 'max_text_len': 74, 'img_w': 800, 'img_h': 64}

def mapLT(labels):
    return ''.join(list(map(lambda x: char_list[int(x)], labels)))

def mapTL(text):
    return list(map(lambda x: char_list.index(x), text))

class DataGenerator:
    def __init__(self, data, h, w, batch_size, i_len, max_text_len):
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.i_len = i_len
        self.samples = data
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.curr_index = 0
    
    def generate(self):
        self.imgs = np.zeros((self.n, self.h, self.w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = preprocess(img_filepath, self.h, self.w)
            self.imgs[i, :, :] = img
            self.texts.append(text)
    
    def nextSample(self):
        self.curr_index += 1
        if self.curr_index >= self.n:
            self.curr_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.curr_index]], self.texts[self.indexes[self.curr_index]]

    def nextBatch(self):
        while(True):
            if K.image_data_format() == 'channel_first':
                X_data = np.ones([self.batch_size,1,self.img_w,self.img_h])
            else:
                X_data = np.ones([self.batch_size,self.max_text_len])
            Y_data = np.zeros([self.batch_size,self.max_text_len])
            input_length = np.ones((self.batch_size,1)) * self.i_len
            label_length = np.zeros((self.batch_size,1))

            for i in range(self.batch_size):
                img,text = self.nextSample()
                img = img.T
                if K.image_data_format() == 'channel_first':
                    img = np.expand_dims(img,0)
                else:
                    img = np.expand_dims(img,-1)
                X_data[i] = img
                Y_data[i,:len(text)] = mapTL(text)
                label_length[i] = len(text)
            
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }

            outputs = {'ctc':np.zeros([self.batch_size])}
            yield(inputs,outputs)
            