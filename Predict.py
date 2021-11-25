import numpy as np
from Utils import *
from WordSeg import wordSeg, imgPrep
import matplotlib.pyplot as plt
import cv2
from Preprocessor import preprocess
from keras import backend as K
from keras.utils import plot_model
from Spell import correction_list
import shutil
from keras.models import model_from_json

with open('/content/word_model.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights('')
test_img = ''

img = imgPrep(cv2.imread(test_img), 64)
img2 = img.copy()
res = wordSeg(img, kernelSize = 25, sigma = 11, theta = 7, minArea = 100)

if not os.path.exists('tmp'):
    os.mkdir('tmp')

for (j, w) in enumerate(res):
    (wordBox, wordImg) = w
    (x, y, w, h) = wordBox
    cv2.imwrite('tmp/%d.png'%j, wordImg)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0,255), 1)

cv2.imwrite('', img2)
plt.imshow(img2)
imgFiles = os.listdir('tmp')
pred_line = []

for f in imgFiles:
    pred_line.append(predict_image(model, 'tmp' + f, True))

print('******PREDICTION******')
print('MODEL PREDICTION:' + ' '.join(pred_line))
pred_line = correction_list(pred_line)
plt.show()
shutil.rmtree('tmp')