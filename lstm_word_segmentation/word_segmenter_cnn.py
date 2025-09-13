from pathlib import Path
import numpy as np
import os, json
import hypertune
from icu import Char
from keras.layers import Dense, TimeDistributed, Embedding, Dropout, Input, Conv1D, Maximum
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping

from . import constants
from .helpers import sigmoid, save_training_plot, upload_to_gcs
from .text_helpers import get_best_data_text, get_lines_of_text, get_segmented_file_in_one_line
from .accuracy import Accuracy
from .line import Line
from .bies import Bies
from .grapheme_cluster import GraphemeCluster
from .code_point import CodePoint

class WordSegmenterCNN:
    """
    A class that let you make a CNN, train it, and test it.
    Args:
        input_clusters_num: number of top grapheme clusters used to train the model
        input_embedding_dim: length of the embedding vectors for each grapheme cluster
        input_hunits: number of hidden units used in feature layer of CNN
        input_dropout_rate: dropout rate used in layers after the embedding and after the CNN
        input_output_dim: dimension of the output layer
        input_epochs: number of epochs used to train the model
        input_training_data: name of the data used to train the model
        input_evaluation_data: name of the data used to evaluate the model
        input_language: shows what is the language used to train the model (e.g. Thai, Burmese, ...)
        input_embedding_type: determines what type of embedding to be used in the model. Possible values are
        "grapheme_clusters_tf", "grapheme_clusters_man", and "generalized_vectors"
    """
    def __init__(self, input_name, input_clusters_num, input_embedding_dim,
                 input_dropout_rate, input_output_dim, input_epochs, input_training_data, input_evaluation_data,
                 input_language, input_embedding_type, filters, learning_rate):
        self.name = input_name
        self.clusters_num = input_clusters_num
        self.embedding_dim = input_embedding_dim
        self.dropout_rate = input_dropout_rate
        self.output_dim = input_output_dim
        self.epochs = input_epochs
        self.training_data = input_training_data
        self.evaluation_data = input_evaluation_data
        self.language = input_language
        self.embedding_type = input_embedding_type
        self.model = None
        self.filters = filters
        self.learning_rate = learning_rate

        # Constructing the grapheme cluster dictionary -- this will be used if self.embedding_type is Grapheme Clusters
        ratios = None
        if self.language == "Thai":
            if "exclusive" in self.training_data:
                ratios = constants.THAI_EXCLUSIVE_GRAPH_CLUST_RATIO
            else:
                ratios = constants.THAI_GRAPH_CLUST_RATIO
        elif self.language == "Burmese":
            if "exclusive" in self.training_data:
                ratios = constants.BURMESE_EXCLUSIVE_GRAPH_CLUST_RATIO
            else:
                ratios = constants.BURMESE_GRAPH_CLUST_RATIO
        elif self.language == "Thai_Burmese":
            ratios = constants.THAI_BURMESE_GRAPH_CLUST_RATIO
        else:
            print("Warning: the input language is not supported")
        cnt = 0
        self.graph_clust_dic = dict()
        for key in ratios.keys():
            if cnt < self.clusters_num - 1:
                self.graph_clust_dic[key] = cnt
            if cnt == self.clusters_num - 1:
                break
            cnt += 1

        # Loading the codepoints dictionary -- this will be used if self.embedding_type is codepoints
        # If you want to group some of the codepoints into buckets, that code should go here to change
        # self.codepoint_dic appropriately
        if self.language == "Thai":
            self.codepoint_dic = constants.THAI_CODE_POINT_DICTIONARY
        if self.language == "Burmese":
            self.codepoint_dic = constants.BURMESE_CODE_POINT_DICTIONARY
        self.codepoints_num = len(self.codepoint_dic) + 1

        # Constructing the letters dictionary -- this will be used if self.embedding_type is Generalized Vectors
        self.letters_dic = dict()
        if self.language in ["Thai", "Burmese"]:
            smallest_unicode_dec = None
            largest_unicode_dec = None

            # Defining the Unicode box for model's language
            if self.language == "Thai":
                smallest_unicode_dec = int("0E01", 16)
                largest_unicode_dec = int("0E5B", 16)
            elif self.language == "Burmese":
                smallest_unicode_dec = int("1000", 16)
                largest_unicode_dec = int("109F", 16)

            # Defining the codepoint buckets that will get their own individual embedding vector
            # 1: Letters, 2: Marks, 3: Digits, 4: Separators, 5: Punctuations, 6: Symbols, 7: Others
            separate_slot_buckets = []
            separate_codepoints = []
            if self.embedding_type == "generalized_vectors_123":
                separate_slot_buckets = [1, 2, 3]
            elif self.embedding_type == "generalized_vectors_12":
                separate_slot_buckets = [1, 2]
            elif self.embedding_type == "generalized_vectors_12d0":
                separate_slot_buckets = [1, 2]
                if self.language == "Burmese":
                    separate_codepoints = [4160, 4240]
                if self.language == "Thai":
                    separate_codepoints = [3664]
            elif self.embedding_type == "generalized_vectors_125":
                separate_slot_buckets = [1, 2, 5]
            elif self.embedding_type == "generalized_vectors_1235":
                separate_slot_buckets = [1, 2, 3, 5]

            # Constructing letters dictionary
            cnt = 0
            for i in range(smallest_unicode_dec, largest_unicode_dec + 1):
                ch = chr(i)
                if constants.CHAR_TYPE_TO_BUCKET[Char.charType(ch)] in separate_slot_buckets:
                    self.letters_dic[ch] = cnt
                    cnt += 1
            for unicode_dec in separate_codepoints:
                ch = chr(unicode_dec)
                self.letters_dic[ch] = cnt
                cnt += 1

            # After making the letters dictionary, we can call different versions of the generalized vectors same thing
            if "generalized_vectors" in self.embedding_type:
                self.embedding_type = "generalized_vectors"

        else:
            print("Warning: the generalized_vectros embedding type is not supported for this language")

    def data_generator(self, valid=False):
        LENGTH = 200
        if self.training_data in ["BEST", "exclusive BEST", "pseudo BEST"]:
            if valid:
                start, end = 80, 90
            else:
                start, end = 1, 80
                
            if self.training_data == "BEST":
                pseudo, exclusive = False, False
            elif self.training_data == "exclusive BEST":
                pseudo, exclusive = False, True
            else:
                pseudo, exclusive = True, False
            for i in range(start, end):
                text = get_best_data_text(i, i+1, pseudo=pseudo, exclusive=exclusive)
                x_data, y_data = self._get_trainable_data(text)
                if self.embedding_type == 'grapheme_clusters_tf':
                    x, y = np.array([tok.graph_clust_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
                elif self.embedding_type == 'codepoints':
                    x, y = np.array([tok.codepoint_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
                for pos in range(0, len(x)-LENGTH+1, LENGTH):
                    x_chunk = x[pos : pos + LENGTH]
                    y_chunk = y[pos : pos + LENGTH]
                    yield x_chunk, y_chunk
        else:
            if valid:
                if self.training_data == "my":
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_valid.txt')
                    text = get_segmented_file_in_one_line(file, input_type="icu_segmented", output_type="icu_segmented")
                elif self.training_data == "exclusive my":
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_valid_exclusive.txt')
                    text = get_segmented_file_in_one_line(file, input_type="icu_segmented", output_type="icu_segmented")
            else:
                if self.training_data == "my":
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_train.txt')
                    text = get_segmented_file_in_one_line(file, input_type="icu_segmented", output_type="icu_segmented")
                elif self.training_data == "exclusive my":
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_train_exclusive.txt')
                    text = get_segmented_file_in_one_line(file, input_type="icu_segmented", output_type="icu_segmented")
            x_data, y_data = self._get_trainable_data(text)
            if self.embedding_type == 'grapheme_clusters_tf':
                x, y = np.array([tok.graph_clust_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
            elif self.embedding_type == 'codepoints':
                x, y = np.array([tok.codepoint_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
            for pos in range(0, len(x)-LENGTH+1, LENGTH):
                x_chunk = x[pos : pos + LENGTH]
                y_chunk = y[pos : pos + LENGTH]
                yield x_chunk, y_chunk

    def _conv1d_same(self, x, kernel, bias, dilation=1):
        L, Cin = x.shape
        K, _, Cout = kernel.shape
        pad = dilation * (K - 1) // 2
        x_pad = np.zeros((L + 2 * pad, Cin), dtype=x.dtype)
        x_pad[pad:pad + L, :] = x

        y = np.zeros((L, Cout), dtype=x.dtype)
        for i in range(L):
            acc = np.zeros((Cout,), dtype=x.dtype)
            for k in range(K):
                idx = i + k * dilation
                acc += np.matmul(x_pad[idx], kernel[k])
            y[i] = acc + bias
        return y
    
    def _manual_predict(self, test_input):
        """
        Implementation of the tf.predict function manually. This function works for inputs of any length, and only uses
        model weights obtained from self.model.weights.
        Args:
            test_input: the input text
        """
        dtype = np.float32
        embedarr = self.model.weights[0].numpy().astype(dtype)
        if self.embedding_type == "grapheme_clusters_tf":
            x = np.array([embedarr[token.graph_clust_id] for token in test_input])
        elif self.embedding_type == "codepoints":
            x = np.array([embedarr[token.codepoint_id] for token in test_input])
        else:
            print("Warning: this embedding type is not implemented for manual prediction")

        w1, b1 = self.model.weights[1].numpy().astype(dtype), self.model.weights[2].numpy().astype(dtype)
        w2, b2 = self.model.weights[3].numpy().astype(dtype), self.model.weights[4].numpy().astype(dtype)

        y1 = self._conv1d_same(x, w1, b1, dilation=1)
        y1 = np.maximum(0, y1) # ReLU

        y2 = self._conv1d_same(x, w2, b2, dilation=2) # Conv1D
        y2 = np.maximum(0, y2) # ReLU

        maximum = np.maximum(y1, y2)

        w4, b4 = self.model.weights[5].numpy().astype(dtype), self.model.weights[6].numpy().astype(dtype)
        logits = np.matmul(maximum, np.squeeze(w4, 0)) + b4
        
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)

        return probs.astype(dtype)
    
    def segment_arbitrary_line(self, input_line):
        """
        This function uses the CNN model to segment an unsegmented line and compare it to ICU and deepcut.
        Args:
            input_line: the string that needs to be segmented. It is supposed to be unsegmented
        """
        line = Line(input_line, "unsegmented")
        grapheme_clusters_in_line = len(line.char_brkpoints) - 1

        # Using CNN model to segment the input line
        if self.embedding_type == "codepoints":
            x_data = []
            for i in range(len(line.unsegmented)):
                x_data.append(CodePoint(line.unsegmented[i], self.codepoint_dic))
        else:
            x_data = []
            for i in range(grapheme_clusters_in_line):
                char_start = line.char_brkpoints[i]
                char_finish = line.char_brkpoints[i + 1]
                curr_char = line.unsegmented[char_start: char_finish]
                x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))
        y_hat = Bies(input_bies=self._manual_predict(x_data), input_type="mat")

        # Making a pretty version of the output of the CNN, where bars show the boundaries of words
        y_hat_pretty = ""
        if self.embedding_type == "codepoints":
            for i in range(len(line.unsegmented)):
                if y_hat.str[i] in ['b', 's']:
                    y_hat_pretty += "|"
                y_hat_pretty += line.unsegmented[i]
            y_hat_pretty += "|"
        else:
            y_hat_pretty = ""
            for i in range(grapheme_clusters_in_line):
                char_start = line.char_brkpoints[i]
                char_finish = line.char_brkpoints[i + 1]
                curr_char = line.unsegmented[char_start: char_finish]
                if y_hat.str[i] in ['b', 's']:
                    y_hat_pretty += "|"
                y_hat_pretty += curr_char
            y_hat_pretty += "|"

        return y_hat_pretty

    def _get_trainable_data(self, input_line):
        """
        Given a segmented line, generates a list of input data (with respect to the embedding type) and a n*4 np array
        that represents BIES where n is the length of the unsegmented line.
        Args:
            input_line: the unsegmented line
        """
        # Finding word breakpoints
        # Note that it is possible that input is segmented manually instead of icu. However, for both cases we set that
        # input_type equal to "icu_segmented" because that doesn't affect performance of this function. This way we
        # won't need unnecessary if/else for "man_segmented" and "icu_segmented" throughout rest of this function.
        line = Line(input_line, "icu_segmented")

        # x_data and y_data will be codepoint based if self.embedding_type is codepoints
        if self.embedding_type == "codepoints":
            true_bies = line.get_bies_codepoints("icu")
            y_data = true_bies.mat
            line_len = len(line.unsegmented)
            x_data = []
            for i in range(line_len):
                x_data.append(CodePoint(line.unsegmented[i], self.codepoint_dic))
        # x_data and y_data will be grapheme clusters based if self.embedding type is grapheme_clusters or generalized_
        # vectors
        else:
            true_bies = line.get_bies_grapheme_clusters("icu")
            y_data = true_bies.mat
            line_len = len(line.char_brkpoints) - 1
            x_data = []
            for i in range(line_len):
                char_start = line.char_brkpoints[i]
                char_finish = line.char_brkpoints[i + 1]
                curr_char = line.unsegmented[char_start: char_finish]
                x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))

        return x_data, y_data

    def train_model(self):
        """
        This function trains the model using the dataset specified in the __init__ function. It combine all lines in
        the data set with a space between them and then divide this large string into batches of fixed length self.n.
        in reading files, if `pseudo` is True then we use icu segmented text instead of manually segmented texts to
        train the model.
        """
        base = tf.data.Dataset.from_generator(
            lambda: self.data_generator(valid=False),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,4), dtype=tf.int32)
            )
        )
        element_count = int(base.reduce(tf.constant(0, dtype=tf.int64), lambda count, _: count + 1))
        train_dataset = base.cache().shuffle(element_count, reshuffle_each_iteration=True).padded_batch(batch_size=1024, padded_shapes=([None], [None, 4]), padding_values=(-1,0)).prefetch(tf.data.AUTOTUNE)
    
        valid_dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(valid=True),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,4), dtype=tf.int32)
            )
        ).padded_batch(batch_size=1024, padded_shapes=([None], [None,4]), padding_values=(-1,0))

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Building the model
        inp = Input(shape=(None,), dtype="int32")
        if self.embedding_type == "grapheme_clusters_tf":
            x = Embedding(input_dim=self.clusters_num, output_dim=self.embedding_dim)(inp)
        elif self.embedding_type == "grapheme_clusters_man":
            x = TimeDistributed(Dense(input_dim=self.clusters_num, units=self.embedding_dim, use_bias=False,
                                            kernel_initializer='uniform'))(inp)
        elif self.embedding_type == "generalized_vectors":
            x = TimeDistributed(Dense(self.embedding_dim, activation=None, use_bias=False,
                                            kernel_initializer='uniform'))(inp)
        elif self.embedding_type == "codepoints":
            x = Embedding(input_dim=self.codepoints_num, output_dim=self.embedding_dim, input_length=(None,))(inp)
        else:
            print("Warning: the embedding_type is not implemented")
        x = Dropout(self.dropout_rate)(x)
        y1 = Conv1D(filters=self.filters, kernel_size=3, padding="same", activation="relu")(x)
        y2 = Conv1D(filters=self.filters, kernel_size=5, dilation_rate=2, padding="same", activation="relu")(x)
        x = Maximum()([y1, y2])
        out = Conv1D(filters=self.output_dim, kernel_size=1, activation="softmax")(x) 
        model = Model(inp, out)
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # opt = keras.optimizers.SGD(learning_rate=0.4, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], jit_compile=False)
        # Fitting the model
        history = model.fit(train_dataset, epochs=self.epochs, validation_data=valid_dataset, callbacks=[early_stop])
        # Optional hyperparameter tuning
        hp_metric = history.history['val_accuracy'][-1]
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=hp_metric,
            global_step=self.epochs
        )
        save_training_plot(history, Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name))
        self.model = model

    def _test_text_line_by_line(self, file, line_limit, verbose):
        """
        This function tests the model fitted in self.train() line by line, using the lines in file. These lines must be
        already segmented so we can compute the performance of model.
        Args:
            file: the address of the file that is going to be tested
            line_limit: number of lines to be tested. If set to -1, all lines will be tested.
            verbose: determines if we want to show results line by line
        """
        lines = get_lines_of_text(file, "man_segmented")
        if len(lines) < line_limit:
            print("Warning: not enough lines in the test file")
        accuracy = Accuracy()
        for line in lines:
            x_data, y_data = self._get_trainable_data(line.man_segmented)
            y_hat = Bies(input_bies=self._manual_predict(x_data), input_type="mat")
            y_hat.normalize_bies()
            # Updating overall accuracy using the new line
            actual_y = Bies(input_bies=y_data, input_type="mat")
            accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)
        if verbose:
            print("The BIES accuracy (line by line) for file {} : {:.3f}".format(file, accuracy.get_bies_accuracy()))
            print("The F1 score (line by line) for file {} : {:.3f}".format(file, accuracy.get_f1_score()))
        return accuracy

    def test_model_line_by_line(self, verbose, fast=False):
        """
        This function uses the evaluating data to test the model line by line.
        Args:
            verbose: determines if we want to see the the accuracy of each text that is being tested.
            fast: determines if we use small amount of text to run the test or not.
        """
        line_limit = -1
        if fast:
            line_limit = 1000
        accuracy = Accuracy()
        if self.evaluation_data in ["BEST", "exclusive BEST"]:
            texts_range = range(40, 60)
            if fast:
                texts_range = range(90, 97)
            category = ["news", "encyclopedia", "article", "novel"]
            for text_num in texts_range:
                if verbose:
                    print("testing text {}".format(text_num))
                for cat in category:
                    text_num_str = "{}".format(text_num).zfill(5)
                    file = None
                    if self.evaluation_data == "BEST":
                        file = Path.joinpath(Path(__file__).parent.parent.absolute(),
                                             "Data/Best/{}/{}_".format(cat, cat) + text_num_str + ".txt")
                    elif self.evaluation_data == "exclusive BEST":
                        file = Path.joinpath(Path(__file__).parent.parent.absolute(),
                                             "Data/exclusive_Best/{}/{}_".format(cat, cat) + text_num_str + ".txt")
                    text_acc = self._test_text_line_by_line(file=file, line_limit=-1, verbose=verbose)
                    accuracy.merge_accuracy(text_acc)

        elif self.evaluation_data == "SAFT_Thai":
            if self.language != "Thai":
                print("Warning: the current SAFT data is in Thai and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT/test.txt')
            text_acc = self._test_text_line_by_line(file=file, line_limit=-1, verbose=verbose)
            accuracy.merge_accuracy(text_acc)
        elif self.evaluation_data == "my":
            if self.language != "Burmese":
                print("Warning: the my data is in Burmese and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test_segmented.txt')
            text_acc = self._test_text_line_by_line(file=file, line_limit=line_limit, verbose=verbose)
            accuracy.merge_accuracy(text_acc)
        elif self.evaluation_data == "exclusive my":
            if self.language != "Burmese":
                print("Warning: the exvlusive my data is in Burmese and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test_segmented_exclusive.txt')
            text_acc = self._test_text_line_by_line(file=file, line_limit=line_limit, verbose=verbose)
            accuracy.merge_accuracy(text_acc)
        elif self.evaluation_data == "SAFT_Burmese":
            if self.language != "Burmese":
                print("Warning: the my.text data is in Burmese and you are testing a model in another language")
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT_burmese_test.txt')
            text_acc = self._test_text_line_by_line(file=file, line_limit=line_limit, verbose=verbose)
            accuracy.merge_accuracy(text_acc)

        elif self.evaluation_data == "BEST_my":
            if self.language != "Thai_Burmese":
                print("Warning: the current data should be used only for Thai_Burmese multilingual models")
            # Testing for BEST
            acc1 = Accuracy()
            category = ["news", "encyclopedia", "article", "novel"]
            for text_num in range(40, 45):
                print("testing text {}".format(text_num))
                for cat in category:
                    text_num_str = "{}".format(text_num).zfill(5)
                    file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Data/Best/{}/{}_".format(cat, cat)
                                         + text_num_str + ".txt")
                    text_acc = self._test_text_line_by_line(file=file, line_limit=-1, verbose=verbose)
                    acc1.merge_accuracy(text_acc)
            if verbose:
                print("The BIES accuracy by test_model_line_by_line function (Thai): {:.3f}".
                      format(acc1.get_bies_accuracy()))
                print("The F1 score by test_model_line_by_line function (Thai): {:.3f}".format(acc1.get_f1_score()))
            # Testing for my
            file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_test_segmented.txt')
            acc2 = self._test_text_line_by_line(file, line_limit=line_limit, verbose=verbose)
            if verbose:
                print("The BIES accuracy by test_model_line_by_line function (Burmese): {:.3f}".
                      format(acc2.get_bies_accuracy()))
                print("The F1 score by test_model_line_by_line function (Burmese): {:.3f}".format(acc2.get_f1_score()))
            accuracy.merge_accuracy(acc1)
            accuracy.merge_accuracy(acc2)
        else:
            print("Warning: no implementation for line by line evaluating this data exists")
        if verbose:
            print("The BIES accuracy by test_model_line_by_line function: {:.3f}".format(accuracy.get_bies_accuracy()))
            print("The F1 score by test_model_line_by_line function: {:.3f}".format(accuracy.get_f1_score()))
        return accuracy

    def save_cnn_model(self):
        """
        This function saves the current trained model of this word_segmenter instance.
        """
        # Save the model using Keras
        model_path = (Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name))
        tf.saved_model.save(self.model, model_path)

        file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name + "/weights")
        np.save(str(file), self.model.weights)

        model_paths = (Path.joinpath(Path(__file__).parent.parent.absolute(), f"Models/{self.name}/model.keras"))
        self.model.save(model_paths)

        json_file = Path.joinpath(Path(__file__).parent.parent.absolute(), "Models/" + self.name + "/weights.json")
        with open(str(json_file), 'w') as wfile:
            output = dict()
            output["model"] = self.name
            if "grapheme_clusters" in self.embedding_type:
                output["dic"] = self.graph_clust_dic
            elif "codepoints" in self.embedding_type:
                if self.language == "Thai":
                    output["dic"] = constants.THAI_CODE_POINT_DICTIONARY
                if self.language == "Burmese":
                    output["dic"] = constants.BURMESE_CODE_POINT_DICTIONARY
            for i in range(len(self.model.weights)):
                dic_model = dict()
                dic_model["v"] = 1
                mat = self.model.weights[i].numpy()
                if i==5:
                    dic_model["dim"] = list(mat.shape)[1:]
                else:
                    dic_model["dim"] = list(mat.shape)
                data = mat.ravel().tolist()
                dic_model["data"] = data
                output["mat{}".format(i+1)] = dic_model
            json.dump(output, wfile)

        if 'AIP_MODEL_DIR' in os.environ:
            upload_to_gcs(model_path, os.environ['AIP_MODEL_DIR'])
    

    def set_model(self, input_model):
        """
        This function set the current model to an input model
        input_model: the input model
        """
        self.model = input_model


def pick_cnn_model(model_name, embedding, train_data, eval_data):
    """
    This function returns a saved word segmentation instance w.r.t input specifics
    Args:
        model_name: name of the model
        embedding: embedding type used to train the model
        train_data: the data set used to train the model
        eval_data: the data set to test the model. Often, it should have the same structure as training data set.
    """
    file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Models/' + model_name)
    model = keras.layers.TFSMLayer(file, call_endpoint='serving_default')

    # Figuring out name of the model
    language = None
    if "Thai" in model_name:
        language = "Thai"
    elif "Burmese" in model_name:
        language = "Burmese"
    if language is None:
        print("This model name is not valid because it doesn't have name of a valid language in it")

    # Letting the user know how this model has been trained
    if "exclusive" in model_name:
        print("Note: model {} has been trained using an exclusive data set. However, if you like you can still test"
              " it by other types of data sets (not recommended).".format(model_name))

    # Figuring out values for different hyper-parameters
    input_clusters_num = model.weights[0].shape[0]
    input_embedding_dim = model.weights[0].shape[1]
    input_hunits = model.weights[5].shape[2]
    input_filters = model.weights[5].shape[1]

    word_segmenter = WordSegmenterCNN(input_name=model_name, input_clusters_num=input_clusters_num, 
                                      input_embedding_dim=input_embedding_dim, input_hunits=input_hunits, 
                                      input_dropout_rate=0.2, input_output_dim=4, input_epochs=15, 
                                      input_training_data=train_data, input_evaluation_data=eval_data,
                                      input_language=language, input_embedding_type=embedding, 
                                      filters=input_filters, layers=2, learning_rate=0.001)
    word_segmenter.set_model(model)
    return word_segmenter
