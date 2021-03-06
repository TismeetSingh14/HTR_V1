from Parameters import *
from Utils import *
import editdistance
from keras.models import model_from_json
from DataGenerator import DataGenerator

test_data = getPathAndTexts('/content/dataset/smalltest.txt', isWords = True)
print('number of test image: ', len(test_data))

with open('/content/dataset/word_model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('/content/iam_words--50--1999.h5')

ed_chars = num_chars = ed_words = num_words = 0
for path, gt_text in test_data:
    pred_text = predict_image(model, path, is_word=True)
    if gt_text!=pred_text:
        ed_words += 1 
    num_words += 1
    ed_chars += editdistance.eval(gt_text, pred_text)
    num_chars += len(gt_text)

print('ED chars: ', ed_chars)
print('ED words: ', ed_words)
print('CER: ', ed_chars / num_chars)
print('WER: ', ed_words / num_words)