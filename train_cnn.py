from lstm_word_segmentation.word_segmenter_cnn import WordSegmenterCNN
from lstm_word_segmentation.helpers import download_from_gcs
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Dataset file on Google Cloud Storage', type=str)
    parser.add_argument('--language', help='Dataset language: Thai/ Burmese', type=str, default="Thai")
    parser.add_argument('--input-type', help='Dataset input type: unsegmented, man_segmented', type=str, default="unsegmented")
    parser.add_argument('--epochs', help = 'Number of epochs', type=int, default=5)
    parser.add_argument('--filters', help = 'Number of filters', type=int, default=128)
    parser.add_argument('--name', help='Model name, follow Model Specifications convention', type=str, default="test")
    parser.add_argument('--embedding', help='Embedding type such as grapheme_clusters_tf or codepoints', type=str, default="codepoints")
    parser.add_argument('--layers', help='Number of parallel CNN layers', type=int, default=2)
    parser.add_argument('--edim', help='Input embedding dimensions', type=int, default=16)
    parser.add_argument('--hunits', help='Number of neurons after convolution layers', type=int, default=23)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.001)
    args = parser.parse_args()
    arguments = args.__dict__
    return arguments

def main(args):
    download_from_gcs(args['path'], 'Data')
    if args['language'] == 'Thai':
        word_segmenter = WordSegmenterCNN(input_name=args['name'], input_n=50, input_t=10000, input_clusters_num=350,
                                    input_embedding_dim=args['edim'], input_hunits=args['hunits'], input_dropout_rate=0.1, input_output_dim=4,
                                    input_epochs=args['epochs'], input_training_data="BEST",
                                    input_evaluation_data="BEST", input_language='Thai', layers=args['layers'],
                                    input_embedding_type=args['embedding'], filters=args['filters'], learning_rate=args['learning_rate'])
    else:
        word_segmenter = WordSegmenterCNN(input_name=args['name'], input_n=5, input_t=500, input_clusters_num=350,
                                input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.1, input_output_dim=4,
                                input_epochs=args['epochs'], input_training_data="my", input_evaluation_data="my",
                                input_language="Burmese", input_embedding_type="grapheme_clusters_tf")
    word_segmenter.train_model()
    word_segmenter.save_cnn_model()
    word_segmenter.test_model_line_by_line(verbose=True, fast=True)

if __name__ == "__main__":
    args = parser_args()
    main(args)