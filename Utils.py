import os
import numpy as np
import itertools
from Parameters import *
from Preprocessor import preprocess
from keras import backend as K


def decodeLabel(out):
    out_best = list(np.argmax(out[0, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(char_list):
            outstr += char_list[c]
    return outstr

def decodeBatch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(char_list):
                outstr += char_list[c]
        ret.append(outstr)
    return ret

def getPathAndTexts(partition_split_file, is_words):
    paths_and_texts = []
    
    with open (partition_split_file) as f:
            partition_folder = f.readlines()
    partition_folder = [x.strip() for x in partition_folder]
    
    if is_words:
        with open ('words.txt') as f:
            for line in f:
                if not line or line.startswith('#'): # comment in txt file
                    continue
                line_split = line.strip().split(' ')
                if len(line_split) < 9:
                    continue
                status = line_split[1]
                if status == 'err': # er: segmentation of word can be bad
                    continue

                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join('/data/', label_dir, sub_label_dir, fn)

                gt_text = ' '.join(line_split[8:])
                if len(gt_text)>16:
                    continue

                if sub_label_dir in partition_folder:
                    paths_and_texts.append([img_path, gt_text])
    return paths_and_texts

def predict_image(model_predict, path, is_word):
    if is_word:
        width = word_cfg['img_w']
    else:
        width = line_cfg['img_w']
    img = preprocess(path, width, 64)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decodeLabel(net_out_value)
    return pred_texts