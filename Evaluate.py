from Parameters import *
from Utils import *
from sklearn.model_selection import train_test_split
import editdistance
from Spell import correction
from keras.models import model_from_json
from DataGenerator import DataGenerator

test_data = getPathAndTexts('', isWords = True)
print('number of test image: ', len(test_data))

with open('', 'r') as f:
	model = model_from_json(f.read())
model.load_weights('')

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