from lstm_word_segmentation.word_segmenter_cnn import WordSegmenterCNN
from lstm_word_segmentation.word_segmenter_cnn import pick_lstm_model, pick_cnn_model
import time

# Use Bayesian optimization to decide on values of hunits and embedding_dim
'''
bayes_optimization = LSTMBayesianOptimization(input_language="Thai", input_n=50, input_t=10000, input_epochs=1,
                                              input_embedding_type='grapheme_clusters_tf', input_clusters_num=350,
                                              input_training_data="BEST", input_evaluation_data="BEST",
                                              input_hunits_lower=4, input_hunits_upper=64, input_embedding_dim_lower=4,
                                              input_embedding_dim_upper=64, input_c=0.05, input_iterations=2)
bayes_optimization.perform_bayesian_optimization()
'''

# Train a new model -- choose name cautiously to not overwrite other models
# model_name = "Thai_graphclust_2_32_option6"
# word_segmenter = pick_cnn_model(model_name=model_name, embedding="grapheme_clusters_tf",
#                                  train_data="BEST", eval_data="BEST")
# word_segmenter.benchmark_inference("คิดว่าอักษรไทยมีพื้นฐานมาจากอักษรเขมรเก่าซึ่งมีอายุตั้งแต่คริสตศักราช 611 จารึกภาษาไทยที่เก่าแก่ที่สุดปรากฏเมื่อประมาณ พ.ศ. 1292 ตามประเพณีอักษรไทยถูกสร้างขึ้นโดยพ่อขุนรามคำแหงมหาราช"*10000, 1)

# model_name = "Cantonese_codepoint_2_64"
model_name = "Thai_codepoints_2_32_nobn"
word_segmenter = pick_cnn_model(model_name=model_name, embedding="codepoints",
                                 train_data="BEST", eval_data="BEST")
# word_segmenter_lstm.benchmark_inference("คิดว่าอักษรไทยมีพื้นฐานมาจากอักษรเขมรเก่าซึ่งมีอายุตั้งแต่คริสตศักราช 611 จารึกภาษาไทยที่เก่าแก่ที่สุดปรากฏเมื่อประมาณ พ.ศ. 1292 ตามประเพณีอักษรไทยถูกสร้างขึ้นโดยพ่อขุนรามคำแหงมหาราช"*10000, 1)
# word_segmenter = WordSegmenterCNN(input_name=model_name, input_n=200, input_t=10000, input_clusters_num=350,
#                                input_embedding_dim=128, input_hunits=80, input_dropout_rate=0.1, input_output_dim=4,
#                                input_epochs=60, input_training_data="BEST",
#                                input_evaluation_data="BEST", input_language="Cantonese",
#                                input_embedding_type="radicals", filters=128, layers=2, learning_rate=0.003)
# word_segmenter.train_model()
# word_segmenter.save_cnn_model()

#word_segmenter.test_model_line_by_line(verbose=True)


# Choose one of the saved models to use
# '''

# word_segmenter.benchmark_inference("ทำสิ่งต่างๆ ได้มากขึ้นขณะที่อุปกรณ์ล็อกและชาร์จอยู่ด้วยโหมดแอมเบียนท์"*1000, 1)
# print("model_name = {}, embedding dim = {}, hunits = {}".format(word_segmenter.name, word_segmenter.embedding_dim,
#                                                                 word_segmenter.hunits))
# # word_segmenter.save_model()
word_segmenter.test_model_line_by_line(verbose=True, fast=True)
# '''
