import numpy as np
import matplotlib.pyplot as plt
import cv2

def padding(img, h, w, h_, w_):
    h1 = int((h_-h)/2)
    h2 = int((h_-h)/2) + h
    w1 = int((w_-w)/2)
    w2 = int((w_-w)/2) + w
    img_pad = np.ones([h_,w_,3])*255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad

def resizeImage(img, h_, w_):
    h, w = img.shape[:2]
    if w < w_ and h < h_:
        img = padding(img, h, w, h_, w_)
    
    elif w >= w_ and h < h_:
        new_w = w_
        new_h = int(h*new_w/w)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = padding(new_img, new_h, new_w, h_, w_)
    
    elif w < w_ and h >= h_:
        new_h = h_
        new_w = int(w*new_h/h)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = padding(new_img, new_h, new_w, h_, w_)
    
    else:
        r = max(w/w_, h/h_)
        new_w = max(min(w_, int(w / r)), 1)
        new_h = max(min(h_, int(h / r)), 1)
        new_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
        img = padding(new_img, new_h, new_w, h_, w_)
    
    return img

def preprocess(path, h, w):
    img = cv2.imread(path)
    img = resizeImage(img, h, w)
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img = img/255
    return img