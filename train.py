from lstm_word_segmentation.word_segmenter_cnn import WordSegmenterCNN
from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.helpers import download_from_gcs
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Dataset file on Google Cloud Storage', type=str)
    parser.add_argument('--language', help='Dataset language: Thai/ Burmese', type=str, default="Thai")
    parser.add_argument('--input-type', help='Dataset input type: BEST, my', type=str, default="BEST")
    parser.add_argument('--model-type', help='Model type: cnn, lstm', type=str, default="cnn")
    parser.add_argument('--epochs', help = 'Number of epochs', type=int, default=5)
    parser.add_argument('--filters', help = 'Number of filters', type=int, default=128)
    parser.add_argument('--name', help='Model name, follow Model Specifications convention', type=str, default="test")
    parser.add_argument('--embedding', help='Embedding type such as grapheme_clusters_tf or codepoints', type=str, default="codepoints")
    parser.add_argument('--edim', help='Input embedding dimensions', type=int, default=16)
    parser.add_argument('--hunits', help='Number of neurons after convolution layers', type=int, default=23)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=0.001)
    args = parser.parse_args()
    arguments = args.__dict__
    return arguments

def main(args):
    download_from_gcs(args['path'], 'Data')
    if args['model_type'] == 'cnn':
        word_segmenter = WordSegmenterCNN(input_name=args['name'], input_clusters_num=350, input_embedding_dim=args['edim'], 
                                        input_dropout_rate=0.1, input_output_dim=4, input_epochs=args['epochs'], 
                                        input_training_data=args['input_type'], input_evaluation_data=args['input_type'], 
                                        input_language=args['language'], input_embedding_type=args['embedding'], 
                                        filters=args['filters'], learning_rate=args['learning_rate'])
        word_segmenter.train_model()
        word_segmenter.save_cnn_model()
        word_segmenter.test_model_line_by_line(verbose=True, fast=True)
    elif args['model_type'] == 'lstm':
        word_segmenter = WordSegmenter(input_name=args['name'], input_n=200, input_t=600000, input_clusters_num=350,
                                       input_embedding_dim=args['edim'], input_hunits=args['hunits'], input_dropout_rate=0.2, input_output_dim=4,
                                       input_epochs=args['epochs'], input_training_data=args['input_type'], input_evaluation_data=args['input_type'],
                                       input_language=args['language'], input_embedding_type=args['embedding'])
        word_segmenter.train_model()
        word_segmenter.save_model()
        word_segmenter.test_model_line_by_line(verbose=True, fast=True)
    else:
        print("Warning: the input model type is not supported")
        
if __name__ == "__main__":
    args = parser_args()
    main(args)