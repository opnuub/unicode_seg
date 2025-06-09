from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.helpers import download_from_gcs
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help = 'Dataset file on Google Cloud Storage', type = str)
    parser.add_argument('--language', help = 'Dataset language: Thai/ Burmese', type = str, default = "Thai")
    parser.add_argument('--input-type', help = 'Dataset input type: unsegmented, man_segmented', type = str, default = "unsegmented")
    parser.add_argument('--epochs', help = 'Number of epochs', type = int, default = 5)
    parser.add_argument('--name', help = 'Model name, follow Model Specifications convention', type = str, default = "test")
    args = parser.parse_args()
    arguments = args.__dict__
    return arguments

def main(args):
    download_from_gcs(args['path'], 'Data')
    if args['language'] == 'Thai':
        word_segmenter = WordSegmenter(input_name=args['name'], input_n=50, input_t=10000, input_clusters_num=350,
                                    input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                                    input_epochs=args['epochs'], input_training_data="BEST",
                                    input_evaluation_data="BEST", input_language='Thai',
                                    input_embedding_type="codepoints")
    else:
        word_segmenter = WordSegmenter(input_name=args['name'], input_n=5, input_t=500, input_clusters_num=350,
                                input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.2, input_output_dim=4,
                                input_epochs=args['epochs'], input_training_data="my", input_evaluation_data="my",
                                input_language="Burmese", input_embedding_type="codepoints")
    word_segmenter.train_model()
    word_segmenter.save_model()

if __name__ == "__main__":
    args = parser_args()
    main(args)