from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.word_segmenter_cnn import pick_cnn_model
import time, os
from icu import BreakIterator, Locale
# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Thai", input_n=50, input_t=10000, input_epochs=1,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="BEST", input_evaluation_data="BEST",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=2)
bayes_optimization.perform_bayesian_optimization()
'''

seg_text = "|เพราะ|เพียง|กรด|นิวคลีอิค|ของ|ไวรัส|อย่าง|เดียว|ก็|สร้าง|ไวรัส|สมบูรณ์|"
# text = ''.join(seg_text.split('|'))
text = "ปัญหาความแตกต่างที่เกิดขึ้นระหว่างความเป็นธรรมในทางสังคมกับความเป็นธรรมทางกฎหมาย"
# Train a new model -- choose name cautiously to not overwrite other models
model_name = "Thai_codepoints_2_32_0"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# word_segmenter = WordSegmenterCNN(input_name=model_name, input_clusters_num=350, input_embedding_dim=40, input_hunits=32, 
#                                   input_dropout_rate=0.1, input_output_dim=4, input_epochs=100, input_training_data="BEST",
#                                   input_evaluation_data="BEST", input_language="Thai", input_embedding_type="codepoints", 
#                                   filters=32, layers=2, learning_rate=0.001)
# word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=10000, input_clusters_num=350,
#                                input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
#                                input_epochs=10, input_training_data="BEST",
#                                input_evaluation_data="BEST", input_language="Thai",
#                                input_embedding_type="codepoints")
# word_segmenter.train_model()
# word_segmenter.save_model()
#hihere
# word_segmenter = pick_cnn_model(model_name=model_name, embedding="codepoints",
#                                  train_data="BEST", eval_data="BEST")
# word_segmenter.test_model_line_by_line(verbose=True, fast=True)
# input()
# print(text)
# text = "".join(text.split(""))
# print(len(text))
# print(word_segmenter.segment_arbitrary_line(text))
#hihere
# model_name = "Thai_codepoints_exclusive_model4_heavy"
# word_segmenters = pick_lstm_model(model_name=model_name, embedding="codepoints",
#                                  train_data="BEST", eval_data="BEST")
# print(text)
# print(seg_text)
# print(word_segmenter.segment_arbitrary_line(text))
# print(word_segmenters.segment_arbitrary_line(text))
import numpy as np
import numpy as np

def conv1d_same(x, kernel, bias, dilation=1):
    L, Cin = x.shape
    K, _, Cout = kernel.shape
    pad = dilation * (K - 1) // 2
    
    # Pre-reshape kernel to match the Rust linear indexing: [K, Cin, Cout]
    # In Rust, w_row is &w_ki[c * cout..(c + 1) * cout]
    kernel_reshaped = kernel.reshape(K, Cin, Cout)
    
    y = np.zeros((L, Cout), dtype=np.float32)
    
    for t in range(L):
        # Equivalent to 'let mut acc = vec![0.0f32; cout]'
        acc = np.zeros(Cout, dtype=np.float32)
        
        for ki in range(K):
            # Same boundary logic as Rust 'let s = ...'
            s = t + (ki * dilation) - pad
            if 0 <= s < L:
                x_src = x[s] # MatrixBorrowed<1>
                
                for c in range(Cin):
                    xi = x_src[c]
                    w_row = kernel_reshaped[ki, c, :]
                    
                    # GEMV Optimization: Scalar-Vector multiplication
                    # This replicates the Rust loop: acc[o] += xi * w_row[o]
                    # To match Rust's unrolling precision precisely, we use a loop
                    # instead of np.dot or np.matmul.
                    for o in range(Cout):
                        acc[o] += xi * w_row[o]
        
        # Bias addition and ReLU: 'let v = acc[o] + bias[o]; if v > 0.0 { v } else { 0.0 }'
        for o in range(Cout):
            v = acc[o] + bias[o]
            y[t, o] = v if v > 0.0 else 0.0
            
    return y


def r(t, dtype=np.float32):
    # reshape a 1D flat list using its "dim"
    return np.asarray(t["data"], dtype=dtype).reshape(tuple(t["dim"]))

def manual_predict(model, test_input, x, w1, b1, w2, b2, w4, b4):
    
    """
    Implementation of the tf.predict function manually. This function works for inputs of any length, and only uses
    model weights obtained from self.model.weights.
    Args:
        test_input: the input text
    """
    # test_input = "".join([i.codepoint for i in test_input])
    # return self.bies_from_words(parser.parse(test_input))
    # print(test_input)
    # input()
    dtype = np.float32

    y1 = conv1d_same(x, w1, b1, dilation=1)
    y1 = np.maximum(0, y1) # ReLU

    y2 = conv1d_same(x, w2, b2, dilation=2) # Conv1D
    y2 = np.maximum(0, y2) # ReLU

    maximum = np.maximum(y1, y2)

    logits = np.matmul(maximum, w4) + b4
    
    # logits -= logits.max(axis=-1, keepdims=True)
    # probs = np.exp(logits)
    # probs /= probs.sum(axis=-1, keepdims=True)

    return logits.astype(dtype)
print('_'.join(seg_text.split('|')))
# import budoux

# parser = budoux.load_default_thai_parser()
import json
with open('grapheme10.json') as f:
    model = json.load(f)
from budoux.parser_thai import ThaiParser
# from budoux.parser import Parser
parser = ThaiParser(model)
# parser = Parser(model)
model = None
with open('Models/Thai_codepoints_2_32_0/weights.json') as f:
    model = json.load(f)

dtype = np.float32
embedarr  = r(model["mat1"], dtype)          # (V, C)
w1 = r(model["mat2"], dtype)          # (K, C, C1) or similar
b1 = r(model["mat3"], dtype)
w2 = r(model["mat4"], dtype)          # (K, C, C2)
b2 = r(model["mat5"], dtype)
w4 = r(model["mat6"], dtype)          # (C_h, N) or (1, C_h, N)
b4 = r(model["mat7"], dtype)
letters = np.array(list("0011"))
y_pred = ""
y_test = []
budoux_y_pred = []
lstm_y_pred = []
count = 0
def split_by_empty(input_list):
    result = []
    sublist = []
    for item in input_list:
        if item == '':
            if sublist:  # Only add if sublist isn't empty (handles consecutive '')
                result.append(sublist)
                sublist = []
        else:
            sublist.append(item)
    if sublist:  # Add the final remaining sublist
        result.append(sublist)
    return result

with open('new_test.txt', 'r') as f:
    texts = f.readlines()
    t = 0
    print(len(texts))
    input()
    for text in texts:
        t += 1
        print(t)
        s = text.strip('\n').split('|')
        if '' not in s:
            y_test += s
            raw_text = ''.join(s)
            budx = parser.parse(raw_text)
            budoux_y_pred += budx

            # ids = [model["dic"].get(ch, 73) for ch in raw_text]
            # x = np.array([embedarr[i] for i in ids], dtype=np.float32)
            # logits = manual_predict(model, raw_text, x, w1, b1, w2, b2, w4, b4)
            # idx = logits.argmax(axis=1)
            # out = "".join(letters[idx])[:-1] + "1"
            # y_pred += out
        else:
            lst = split_by_empty(s)
            for sublist in lst:
                y_test += sublist
                raw_text = ''.join(sublist)
                budx = parser.parse(raw_text)
                budoux_y_pred += budx

                # ids = [model["dic"].get(ch, 73) for ch in raw_text]
                # x = np.array([embedarr[i] for i in ids], dtype=np.float32)
                # logits = manual_predict(model, raw_text, x, w1, b1, w2, b2, w4, b4)
                # idx = logits.argmax(axis=1)
                # out = "".join(letters[idx])[:-1] + "1"
                # y_pred += out
        # lstm = word_segmenters.segment_arbitrary_line(raw_text).split('|')[:-1]
        # if lstm[0] == '':
        #     lstm_y_pred += lstm[1:]
        # else:
        #     lstm_y_pred += lstm

cnn_y_pred = []
word = ""
for i in y_pred:
    word += i
    if i=="1":
        cnn_y_pred.append(word)
        word = ""
from metrics import MultiClassMetrics
# print('CNN')
# metrics = MultiClassMetrics(y_test, cnn_y_pred, 'bies')
# print(metrics.accuracy)
# print('Weighted:', metrics.get_f1_score(True), 'Macro:', metrics.get_f1_score(False), '\n')
# print(metrics.get_f1_score(False))

print('BudouX')
metrics = MultiClassMetrics(y_test, budoux_y_pred, 'bies')
print(metrics.accuracy)
print('Weighted:', metrics.get_f1_score(True), 'Macro:', metrics.get_f1_score(False), '\n')

# print('LSTM')
# metrics = MultiClassMetrics(y_test, lstm_y_pred, 'bies')
# print(metrics.accuracy)
# print('Weighted:', metrics.get_f1_score(True), 'Macro:', metrics.get_f1_score(False), '\n')

# start = time.time()
# for _ in range(1):
#     word_segmenters.segment_arbitrary_line(text)
# print(f'LSTM: {time.time()-start}')
# from icu import BreakIterator, Locale

# words_break_iterator = BreakIterator.createWordInstance(Locale.getRoot())
# start = time.time()
# for _ in range(10):
#     words_break_iterator.setText(text)
#     for brkpoint in words_break_iterator:
#         continue
# print(f'Dict: {time.time()-start}')
# start = time.time()
# for _ in range(10):
#     parser.parse(text)
# print(f'BudouX: {time.time()-start}')

# print(parser.parse("มิสเตอร์มัมเบิลส์ทั้งสามก้มลงฟังอย่างตั้งใจ"))
# model_name = "Burmese_temp"
# word_segmenter = pick_cnn_model(model_name=model_name, embedding="codepoints",
#                                  train_data="my", eval_data="my")
# # print(word_segmenter.segment_arbitrary_line(text))
# word_segmenter.test_model_line_by_line(verbose=True, fast=True)

"""
from lstm_word_segmentation.lstm_bayesian_optimization import LSTMBayesianOptimization
from lstm_word_segmentation.word_segmenter import pick_lstm_model
from lstm_word_segmentation.word_segmenter_cnn import WordSegmenterCNN


# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Burmese", input_n=50, input_t=10000, input_epochs=2,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="my", input_evaluation_data="my",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=1)
bayes_optimization.perform_bayesian_optimization()
'''

# Train a new model -- choose name cautiously to not overwrite other models
model_name = "Burmese_temp"
word_segmenter = WordSegmenterCNN(input_name=model_name, input_clusters_num=350,
                               input_embedding_dim=40, input_hunits=27, input_dropout_rate=0.1, input_output_dim=4,
                               input_epochs=200, input_training_data="my", input_evaluation_data="my",
                               input_language="Burmese", layers=2, input_embedding_type="codepoints",
                               filters=32, learning_rate=0.001)
word_segmenter.train_model()
word_segmenter.save_cnn_model()
# word_segmenter.test_model_line_by_line(verbose=True)
print(word_segmenter.segment_arbitrary_line("ဘုရင်ကစိတ်မညစ်ပါနဲ့၊ငါ့ကိုဒီလိုမကြည့်နဲ့"))
# Choose one of the saved models to use
# '''
# word_segmenter = pick_lstm_model(model_name="Burmese_graphclust_model5_heavy", embedding="grapheme_clusters_tf",
#                                  train_data="my", eval_data="my")
# print("model_name = {}, embedding dim = {}, hunits = {}".format(word_segmenter.name, word_segmenter.embedding_dim,
#                                                                 word_segmenter.hunits))
# word_segmenter.save_model()
# word_segmenter.test_model_line_by_line(verbose=True, fast=True)
# '''

"""