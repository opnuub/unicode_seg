from lstm_word_segmentation.word_segmenter import WordSegmenter
from lstm_word_segmentation.helpers import download_from_gcs
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    help = 'Dataset file on Google Cloud Storage',
    type = str
)
parser.add_argument(
    '--language',
    help = 'Dataset language: Thai/ Burmese',
    type = str,
)
parser.add_argument(
    '--output',
    help = 'Directory to output model artifacts',
    type = str,
    default = os.environ['AIP_MODEL_DIR'] if 'AIP_MODEL_DIR' in os.environ else ""
)
parser.add_argument(
    '--input-type',
    help = 'Dataset input type: unsegmented, man_segmented',
    type = str,
    default = "unsegmented"
)
args = parser.parse_args()
arguments = args.__dict__

model_name = "test"
download_from_gcs(arguments['path'], 'Data')
if arguments['language'] == 'Thai':
    word_segmenter = WordSegmenter(input_name=model_name, input_n=50, input_t=10000, input_clusters_num=350,
                                input_embedding_dim=16, input_hunits=23, input_dropout_rate=0.2, input_output_dim=4,
                                input_epochs=5, input_training_data="BEST",
                                input_evaluation_data="BEST", input_language='Thai',
                                input_embedding_type="codepoints")
else:
    word_segmenter = WordSegmenter(input_name=model_name, input_n=200, input_t=600000, input_clusters_num=350,
                               input_embedding_dim=28, input_hunits=14, input_dropout_rate=0.2, input_output_dim=4,
                               input_epochs=20, input_training_data="my", input_evaluation_data="my",
                               input_language="Burmese", input_embedding_type="codepoints")
word_segmenter.train_model()
word_segmenter.save_model()