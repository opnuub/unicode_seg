from pathlib import Path
import numpy as np
import json, os, time, io
import hypertune
from icu import Char
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Embedding, Dropout, Input, Conv1D, BatchNormalization, ReLU, Maximum
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping

from . import constants
from .helpers import sigmoid, save_training_plot, upload_to_gcs
from .text_helpers import get_segmented_file_in_one_line, get_best_data_text, get_lines_of_text, get_hkcancor_text
from .accuracy import Accuracy
from .line import Line
from .bies import Bies
from .grapheme_cluster import GraphemeCluster
from .code_point import CodePoint
from .chinese import Radicals

class KerasBatchGenerator(object):
    """
    A batch generator component, which is used to generate batches for training, validation, and evaluation.
    Args:
        x_data: A list of GraphemeCluster objects that is the input of the model
        y_data: A np array that contains output of the model
        n: length of the input and output in each batch
        batch_size: number of batches
    """
    def __init__(self, x_data, y_data, n, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.batch_size = batch_size
        self.dim_output = self.y_data.shape[1]
        if len(x_data) != y_data.shape[0]:
            print("Warning: x_data and y_data have not compatible sizes!")
        if len(x_data) < batch_size * n:
            print("Warning: x_data or y_data is not large enough!")

    def generate(self, embedding_type):
        """
        This function generates batches used for training and validation
        """
        x, y = self.generate_once(embedding_type)
        while True:
            yield x, y

    def generate_once(self, embedding_type):
        """
        This function generates batches only once and is used for testing
        """
        y = np.zeros([self.batch_size, self.n, self.dim_output])
        x = None
        if embedding_type == "grapheme_clusters_tf":
            x = np.zeros([self.batch_size, self.n])
        elif embedding_type == "grapheme_clusters_man":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].num_clusters])
        elif embedding_type == "generalized_vectors":
            x = np.zeros([self.batch_size, self.n, self.x_data[0].generalized_vec_length])
        elif embedding_type == "codepoints":
            x = np.zeros([self.batch_size, self.n])
        else:
            print("Warning: the embedding type is not valid")
        for i in range(self.batch_size):
            for j in range(self.n):
                if embedding_type == "grapheme_clusters_tf":
                    x[i, j] = self.x_data[self.n*i + j].graph_clust_id
                if embedding_type == "grapheme_clusters_man":
                    x[i, j, :] = self.x_data[self.n*i + j].graph_clust_vec
                if embedding_type == "generalized_vectors":
                    x[i, j, :] = self.x_data[self.n*i + j].generalized_vec
                if embedding_type == "codepoints":
                    x[i, j] = self.x_data[self.n * i + j].codepoint_id
            y[i, :, :] = self.y_data[self.n * i: self.n * (i + 1), :]
        print(x)
        return x, y


class WordSegmenterCNN:
    """
    A class that let you make a bi-directional LSTM, train it, and test it.
    Args:
        input_n: Length of the input for LSTM model
        input_t: The total length of data used to train and validate the model. It is equal to number of batches times n
        input_clusters_num: number of top grapheme clusters used to train the model
        input_embedding_dim: length of the embedding vectors for each grapheme cluster
        input_hunits: number of hidden units used in each cell of LSTM
        input_dropout_rate: dropout rate used in layers after the embedding and after the bidirectional LSTM
        input_output_dim: dimension of the output layer
        input_epochs: number of epochs used to train the model
        input_training_data: name of the data used to train the model
        input_evaluation_data: name of the data used to evaluate the model
        input_language: shows what is the language used to train the model (e.g. Thai, Burmese, ...)
        input_embedding_type: determines what type of embedding to be used in the model. Possible values are
        "grapheme_clusters_tf", "grapheme_clusters_man", and "generalized_vectors"
    """
    def __init__(self, input_name, input_n, input_t, input_clusters_num, input_embedding_dim, input_hunits,
                 input_dropout_rate, input_output_dim, input_epochs, input_training_data, input_evaluation_data,
                 input_language, input_embedding_type, filters, layers, learning_rate):
        self.name = input_name
        self.n = input_n
        self.t = input_t
        if self.t % self.n != 0:
            print("Warning: t is not divided by n")
        self.clusters_num = input_clusters_num
        # batch_size is the number of batches used in each iteration of back propagation to update model weights
        # The default value is self.t/self.n, but it can be set to other values as well. The only constraint is that
        # self.t should always be greater than self.batch_size * self.n
        self.batch_size = self.t // self.n
        self.embedding_dim = input_embedding_dim
        self.hunits = input_hunits
        self.dropout_rate = input_dropout_rate
        self.output_dim = input_output_dim
        self.epochs = input_epochs
        self.training_data = input_training_data
        self.evaluation_data = input_evaluation_data
        self.language = input_language
        self.embedding_type = input_embedding_type
        self.model = None
        self.filters = filters
        self.layers = layers
        self.learning_rate = learning_rate

        # Constructing the grapheme cluster dictionary -- this will be used if self.embedding_type is Grapheme Clusters
        if self.embedding_type == "grapheme_clusters_tf":
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

        # Loading the code points dictionary -- this will be used if self.embedding_type is Code Points
        # If you want to group some of the code points into buckets, that code should go here to change
        # self.codepoint_dic appropriately
        if self.language == "Thai":
            self.codepoint_dic = constants.THAI_CODE_POINT_DICTIONARY
        if self.language == "Burmese":
            self.codepoint_dic = constants.BURMESE_CODE_POINT_DICTIONARY
        if self.language == "Cantonese":
            self.codepoint_dic = constants.CANTON_CODE_POINT_DICTIONARY
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

            # Defining the code point buckets that will get their own individual embedding vector
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

    def data_generator(self, start, end):
        LENGTH = 200
        if self.language == "Thai":
            for i in range(start, end):
                text = get_best_data_text(i, i+1, False, False)
                x_data, y_data = self._get_trainable_data(text)
                if self.embedding_type == 'grapheme_clusters_tf':
                    x, y = np.array([tok.graph_clust_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
                elif self.embedding_type == 'codepoints':
                    x, y = np.array([tok.codepoint_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
                    LENGTH = 400
                for pos in range(0, len(x)-LENGTH+1, LENGTH):
                    x_chunk = x[pos : pos + LENGTH]
                    y_chunk = y[pos : pos + LENGTH]
                    yield x_chunk, y_chunk
        elif self.language == "Cantonese":
            LENGTH = 400
            if end == 1:
                text = get_hkcancor_text(train=True)
            else:
                text = get_hkcancor_text(train=False)
            x_data, y_data = self._get_trainable_data(text)
            x, y = np.array([tok.radical_id for tok in x_data], dtype=np.int32), np.array(y_data, dtype=np.int32)
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
        # No BatchNormalization
        Implementation of the tf.predict function manually. This function works for inputs of any length, and only uses
        model weights obtained from self.model.weights.
        Args:
            test_input: the input text
        """
        dtype = np.float32
        embedarr = self.model.weights[0].numpy().astype(dtype)
        x = []
        for i in range(len(test_input)):
            if self.embedding_type == "grapheme_clusters_tf":
                x.append(embedarr[test_input[i].graph_clust_id])
            elif self.embedding_type == "grapheme_clusters_man":
                continue
            elif self.embedding_type == "generalized_vectors":
                # input_generalized_vec = test_input[i].generalized_vec
                # x_t = input_generalized_vec.dot(embedarr)
                continue
            elif self.embedding_type == "codepoints":
                x.append(embedarr[test_input[i].codepoint_id])
            else:
                print("Warning: this embedding type is not implemented for manual prediction")
        x = np.stack(x, axis=0)

        w1, b1 = self.model.weights[1].numpy().astype(dtype), self.model.weights[2].numpy().astype(dtype) # Conv1D
        w2, b2 = self.model.weights[3].numpy().astype(dtype), self.model.weights[4].numpy().astype(dtype) # Conv1D

        y1 = self._conv1d_same(x, w1, b1, dilation=1)
        y1 = np.maximum(0, y1) # ReLU

        y2 = self._conv1d_same(x, w2, b2, dilation=2) # Conv1D
        y2 = np.maximum(0, y2) # ReLU

        maximum = np.maximum(y1, y2)

        w3, b3 = self.model.weights[5].numpy().astype(dtype), self.model.weights[6].numpy().astype(dtype)
        w3 = np.squeeze(w3, axis=0)

        hunits_out = np.matmul(maximum, w3) + b3 # Feature dense layer (hunits)
        hunits_out = np.maximum(0, hunits_out) # ReLU

        w4, b4 = self.model.weights[7].numpy().astype(dtype), self.model.weights[8].numpy().astype(dtype)
        logits = np.matmul(hunits_out, np.squeeze(w4, 0)) + b4
        
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        # # Combining Forward and Backward layers through dense time-distributed layer
        # timew = self.model.weights[7].numpy().astype(dtype)
        # timeb = self.model.weights[8].numpy().astype(dtype)
        # est = np.zeros([len(test_input), 4], dtype=dtype)
        # for i in range(len(test_input)):
        #     final_h = np.concatenate((all_h_fw[i, :], all_h_bw[i, :]), axis=0)
        #     final_h = final_h.reshape(1, 2 * self.hunits)
        #     curr_est = final_h.dot(timew) + timeb
        #     curr_est = curr_est[0]
        #     curr_est = np.exp(curr_est) / sum(np.exp(curr_est))
        #     est[i, :] = curr_est
        return probs.astype(dtype)

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
        if self.language == "Cantonese":
            line = Line(input_line, "man_segmented")
        else:
            line = Line(input_line, "icu_segmented")

        # x_data and y_data will be code point based if self.embedding_type is codepoints
        if self.embedding_type == "codepoints":
            true_bies = line.get_bies_codepoints("icu")
            y_data = true_bies.mat
            line_len = len(line.unsegmented)
            x_data = []
            for i in range(line_len):
                x_data.append(CodePoint(line.unsegmented[i], self.codepoint_dic))

        elif self.embedding_type == "radicals":
            from cihai.core import Cihai
            c = Cihai()
            if not c.unihan.is_bootstrapped:
                c.unihan.bootstrap()
            true_bies = line.get_bies_codepoints("icu")
            y_data = true_bies.mat
            line_len = len(line.unsegmented)
            x_data = []
            for i in range(line_len):
                char = c.unihan.lookup_char(line.unsegmented[i]).first()
                if char is None:
                    x_data.append(Radicals(1, 301))
                else:
                    first_radical = char.kRSUnicode.split(" ")[0]
                    x_data.append(Radicals(line.unsegmented[i], int(first_radical.split('.')[0])))
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
        # Get training data of length self.t
        # input_str = None
        # if self.training_data == "BEST":
        #     input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=False, exclusive=False)
        # elif self.training_data == "exclusive BEST":
        #     input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=False, exclusive=True)
        # elif self.training_data == "pseudo BEST":
        #     input_str = get_best_data_text(starting_text=1, ending_text=10, pseudo=True, exclusive=False)
        # elif self.training_data == "my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_train.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="unsegmented", output_type="icu_segmented")
        # elif self.training_data == "exclusive my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_train_exclusive.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="unsegmented", output_type="icu_segmented")
        # elif self.training_data == "SAFT_Burmese":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT_burmese_train.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="man_segmented", output_type="man_segmented")
        # elif self.training_data == "BEST_my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Best_my_train.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="man_segmented", output_type="man_segmented")
        # else:
        #     print("Warning: no implementation for this training data exists!")
        # x_data, y_data = self._get_trainable_data(input_str)
        # if self.t > len(x_data):
        #     print("Warning: size of the training data is less than self.t")
        # x_data = x_data[:self.t]
        # y_data = y_data[:self.t, :]
        # train_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size)

        # # Get validation data of length self.t
        # if self.training_data == "BEST":
        #     input_str = get_best_data_text(starting_text=80, ending_text=90, pseudo=False, exclusive=False)
        # elif self.training_data == "exclusive BEST":
        #     input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=False, exclusive=True)
        # elif self.training_data == "pseudo BEST":
        #     input_str = get_best_data_text(starting_text=10, ending_text=20, pseudo=True, exclusive=False)
        # elif self.training_data == "my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_valid.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="unsegmented", output_type="icu_segmented")
        # elif self.training_data == "exclusive my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/my_valid_exclusive.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="unsegmented", output_type="icu_segmented")
        # elif self.training_data == "SAFT_Burmese":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/SAFT_burmese_test.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="man_segmented", output_type="man_segmented")
        # elif self.training_data == "BEST_my":
        #     file = Path.joinpath(Path(__file__).parent.parent.absolute(), 'Data/Best_my_valid.txt')
        #     input_str = get_segmented_file_in_one_line(file, input_type="man_segmented", output_type="man_segmented")
        # else:
        #     print("Warning: no implementation for this validation data exists!")
        # x_data, y_data = self._get_trainable_data(input_str)
        # if self.t > len(x_data):
        #     print("Warning: size of the validation data is less than self.t")
        # x_data = x_data[:self.t]
        # y_data = y_data[:self.t, :]
        # valid_generator = KerasBatchGenerator(x_data, y_data, n=self.n, batch_size=self.batch_size)

        if self.language == "Thai":
            base = tf.data.Dataset.from_generator(
                lambda: self.data_generator(1, 80),
                output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    tf.TensorSpec(shape=(None,4), dtype=tf.int32)
                )
            )
            train_dataset = base.repeat().shuffle(50000, reshuffle_each_iteration=True).padded_batch(batch_size=1024, padded_shapes=([None], [None,4]), padding_values=(-1,0)).prefetch(tf.data.AUTOTUNE)
        
            valid_dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(80, 90),
                output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    tf.TensorSpec(shape=(None,4), dtype=tf.int32)
                )
            ).padded_batch(batch_size=1024, padded_shapes=([None], [None,4]), padding_values=(-1,0))

        elif self.language == "Cantonese":
            base = tf.data.Dataset.from_generator(
                lambda: self.data_generator(1, 1),
                output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    tf.TensorSpec(shape=(None,4), dtype=tf.int32)
                )
            )
            train_dataset = base.padded_batch(batch_size=128, padded_shapes=([None], [None,4]), padding_values=(-1,0)).prefetch(tf.data.AUTOTUNE)

            valid_dataset = tf.data.Dataset.from_generator(
                lambda: self.data_generator(80, 90),
                output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    tf.TensorSpec(shape=(None,4), dtype=tf.int32)
                )
            ).padded_batch(batch_size=128, padded_shapes=([None], [None,4]), padding_values=(-1,0))

        checkpoiont_dir = Path.joinpath(Path(__file__).parent.parent.absolute(), f"Models/{self.name}/checkpoints")
        checkpoiont_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath = str(checkpoiont_dir / "epoch_{epoch:02d}.keras"),
            save_freq = "epoch",
            save_weights_only = False,
            verbose = 0
        )

        early_stop = EarlyStopping(
            monitor="val_loss",      # metric to watch
            patience=5,             # “no-improve” epochs before stopping
            restore_best_weights=True,
            verbose=1                # prints a message when stopping
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
            x = Embedding(input_dim=self.codepoints_num, output_dim=self.embedding_dim, input_length=self.n)(inp)
        elif self.embedding_type == "radicals":
            x = tf.keras.layers.Lambda(
                lambda t: tf.one_hot(tf.cast(t, 'int32'), depth=301)
            )(inp)
        else:
            print("Warning: the embedding_type is not implemented")
        x = Dropout(self.dropout_rate)(x)
        y1 = Conv1D(filters=self.filters, kernel_size=3, padding="same", activation="relu")(x)
        y2 = Conv1D(filters=self.filters, kernel_size=5, dilation_rate=2, padding="same", activation="relu")(x)
        x = Maximum()([y1, y2])
        x = Conv1D(filters=self.hunits, kernel_size=1, activation="relu")(x)
        x = Dropout(self.dropout_rate)(x)
        out = Conv1D(filters=self.output_dim, kernel_size=1, activation="softmax")(x) 
        model = Model(inp, out)
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # opt = keras.optimizers.SGD(learning_rate=0.4, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], jit_compile=False)
        # Fitting the model
        if self.language == "Thai":
            history = model.fit(train_dataset, steps_per_epoch=700, epochs=self.epochs, validation_data=valid_dataset, callbacks=[early_stop, checkpoint_cb])
        elif self.language == "Cantonese":
            history = model.fit(train_dataset, epochs=self.epochs, validation_data=valid_dataset, callbacks=[checkpoint_cb])
        # Optional hyperparameter tuning
        hp_metric = history.history['val_accuracy'][-1]
        param_count = model.count_params()
        size_mb = param_count * 4 / 1_048_576
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=hp_metric,
            global_step=self.epochs
        )
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='model_size',
            metric_value=size_mb,
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
        for line in lines[:line_limit]:
            x_data, y_data = self._get_trainable_data(line.man_segmented)
            if self.embedding_type == "grapheme_clusters_tf":
                x = np.array([[tok.graph_clust_id for tok in x_data]], dtype="float32")
            elif self.embedding_type == "codepoints":
                x = np.array([[tok.codepoint_id for tok in x_data]], dtype="float32")
            # y_hat = self._manual_predict(x_data)
            # input()
            # y_pred = np.squeeze(self.model.predict(x, verbose=0), axis=0)
            # print(y_pred.shape)
            # input()
            y_hat = Bies(input_bies=self._manual_predict(x_data), input_type="mat")
            y_hat.normalize_bies()
            # Updating overall accuracy using the new line
            actual_y = Bies(input_bies=y_data, input_type="mat")
            accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)
        if verbose:
            print("The BIES accuracy (line by line) for file {} : {:.3f}".format(file, accuracy.get_bies_accuracy()))
            print("The F1 score (line by line) for file {} : {:.3f}".format(file, accuracy.get_f1_score()))
        return accuracy

    def benchmark_inference(self, input_str, repeats):
        line = Line(input_str, input_type="unsegmented")
        grapheme_clusters_in_line = len(line.char_brkpoints) - 1
        x_data = []
        for i in range(grapheme_clusters_in_line):
            char_start = line.char_brkpoints[i]
            char_finish = line.char_brkpoints[i + 1]
            curr_char = line.unsegmented[char_start: char_finish]
            x_data.append(GraphemeCluster(curr_char, self.graph_clust_dic, self.letters_dic))
        graph_clust_ids = [tok.graph_clust_id for tok in x_data]
        graph_clust_ids = np.asarray(graph_clust_ids, dtype="int32")
        x_batch = graph_clust_ids[None, :] 
        true_bies = line.get_bies_grapheme_clusters("icu")
        y_data = true_bies.mat
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(1e-3),
            run_eagerly=False,
        )
        y_pred = self.model(x_batch, training=False)
        y_pred = np.squeeze(y_pred, axis=0)
        y_hat = Bies(input_bies=y_pred, input_type="mat")
        y_hat.normalize_bies()
        actual_y = Bies(input_bies=y_data, input_type="mat")
        accuracy = Accuracy()
        accuracy.update(true_bies=actual_y.str, est_bies=y_hat.str)
        print(accuracy.get_bies_accuracy())
        print(accuracy.get_f1_score())
        t0 = time.perf_counter()
        for _ in range(repeats):
            self.model(x_batch, training=False)
        total_time = (time.perf_counter() - t0)

        print(f"{total_time*1000} ms")

    def test_model_line_by_line(self, verbose, fast=False):
        """
        This function uses the evaluating data to test the model line by line.
        Args:
            verbose: determines if we want to see the the accuracy of each text that is being tested.
            fast: determines if we use small amount of text to run the test or not.
        """
        total_time = 0
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

        if 'AIP_MODEL_DIR' in os.environ:
            upload_to_gcs(model_path, os.environ['AIP_MODEL_DIR'])
    

    def set_model(self, input_model):
        """
        This function set the current model to an input model
        input_model: the input model
        """
        import keras
        model_path = (Path.joinpath(Path(__file__).parent.parent.absolute(), f"Models/{self.name}/model.keras"))
        model = keras.saving.load_model(model_path, compile=False)
        self.model = model


def pick_cnn_model(model_name, embedding, train_data, eval_data):
    """
    Load a saved CNN-based word segmentation model and return a WordSegmenter instance.
    Args:
        model_name: Name of the saved model directory (under Models/).
        embedding: Embedding type used to train the model (e.g. "grapheme_clusters_tf", "codepoints", etc.).
        train_data: Dataset name used for training (e.g. "BEST").
        eval_data: Dataset name for evaluation (should correspond to training dataset structure).
    """
    # Determine language from model name
    language = None
    if "Thai" in model_name:
        language = "Thai"
    elif "Burmese" in model_name:
        language = "Burmese"
    else:
        print("Warning: model name does not specify a supported language.")
    
    # Warn if model was trained on exclusive dataset
    if "exclusive" in model_name:
        print(f"Note: model {model_name} was trained on an exclusive dataset.")
    
    input_n = 200
    input_t = 500000
    
    # Infer embedding and model dimensions from weights
    # Weight 0: Embedding matrix of shape (vocab_size, embedding_dim)
    # vocab_size = loaded_layer.weights[0].shape[0]    # number of input tokens (clusters/codepoints)
    # embedding_dim = loaded_layer.weights[0].shape[1]  # embedding vector dimension
    # Last Dense layer weights (just before softmax) have shape (hunits, output_dim)
    output_dim = 4    # should be 4 for BIES tags
    hunits = 23        # hidden units in the TimeDistributed Dense layer
    
    # Create a WordSegmenter instance with these parameters
    word_segmenter = WordSegmenterCNN(
        input_name=model_name,
        input_n=input_n,
        input_t=input_t,
        input_clusters_num=350,       # for grapheme clusters this includes +1 for "unknown"
        input_embedding_dim=200,
        input_hunits=hunits,
        input_dropout_rate=0.2,
        input_output_dim=output_dim,
        input_epochs=1,                     # epochs value not critical for inference
        input_training_data=train_data,
        input_evaluation_data=eval_data,
        input_language=language,
        input_embedding_type=embedding,
        filters=32,
        layers=2,
        learning_rate=0.001
    )
    # Assign the loaded model to the WordSegmenter
    word_segmenter.set_model(None)
    return word_segmenter

def pick_lstm_model(model_name, embedding, train_data, eval_data):
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
    input_hunits = model.weights[1].shape[1]//4
    input_n = None
    input_t = None
    if "genvec" in model_name or "graphclust" in model_name:
        input_n = 50
        input_t = 100000
        if "heavy" in model_name:
            input_n = 200
            input_t = 600000
        elif "heavier" in model_name:
            input_n = 200
            input_t = 2000000
    elif "codepoints" in model_name:
        input_n = 100
        input_t = 200000
        if "heavy" in model_name:
            input_n = 300
            input_t = 1200000
    if input_n is None:
        print("This model name is not valid because it doesn't have name of the embedding type in it")
    word_segmenter = WordSegmenterCNN(input_name=model_name, input_n=input_n, input_t=input_t,
                                   input_clusters_num=input_clusters_num, input_embedding_dim=input_embedding_dim,
                                   input_hunits=input_hunits, input_dropout_rate=0.2, input_output_dim=4,
                                   input_epochs=15, input_training_data=train_data, input_evaluation_data=eval_data,
                                   input_language=language, input_embedding_type=embedding)
    word_segmenter.set_model(model)
    return word_segmenter
