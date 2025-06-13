import numpy as np
from . import constants
from google.cloud import storage
import os

def is_ascii(input_str):
    """
    A very basic function that checks if all elements of str are ASCII or not
    Args:
        input_str: input string
    """
    return all(ord(char) < 128 for char in input_str)


def diff_strings(str1, str2):
    """
    A function that returns the number of elements of two strings that are not identical
    Args:
        str1: the first string
        str2: the second string
    """
    if len(str1) != len(str2):
        print("Warning: length of two strings are not equal")
        return -1
    return sum(str1[i] != str2[i] for i in range(len(str1)))


def sigmoid(inp):
    """
    Computes the sigmoid function of a scalar or a 1d numpy array
    Args:
        inp: the input which can be a scalar or a 1d numpy array
    """
    inp = np.asarray(inp)
    scalar_input = False
    if inp.ndim == 0:
        inp = inp[None]
        scalar_input = True
    # Checking for case when the input is an array/np.array of arrays. In this case only the first element of inp is
    # used. A common example is when A = np.array([np.array([1, 2, 3])]).
    if type(inp[0]) == np.ndarray:
        inp = inp[0]
    out = []
    for x in inp:
        if x < -20:
            out.append(0)
        else:
            out.append(1.0/(1.0 + np.exp(-x)))
    out = np.array(out)
    if scalar_input:
        return np.squeeze(out)
    return out


def print_grapheme_clusters(thrsh, language, exclusive):
    """
    This function print the grapheme clusters and their frequencies for a given langauge. It also computes what
    percentage of grapheme clusters form which percent of the text
    Args:
        thrsh: shows what percent of the text we want to be covered by grapheme clusters
        language: shows the language that we are working with
        exclusive: shows if we only consider grapheme clusters in a single script or not
    """
    ratios = None
    if language == "Thai" and exclusive is False:
        ratios = constants.THAI_GRAPH_CLUST_RATIO
    if language == "Thai" and exclusive is True:
        ratios = constants.THAI_EXCLUSIVE_GRAPH_CLUST_RATIO
    if language == "Burmese" and exclusive is False:
        ratios = constants.BURMESE_GRAPH_CLUST_RATIO
    if language == "Burmese" and exclusive is True:
        ratios = constants.BURMESE_EXCLUSIVE_GRAPH_CLUST_RATIO
    if language == "Thai-Burmese":
        ratios = constants.THAI_BURMESE_GRAPH_CLUST_RATIO
    if ratios is None:
        print("No grapheme cluster dictionary has been computed for the input language.")
        return
    cum_sum = 0
    cnt = 0
    for val in ratios.values():
        cum_sum += val
        cnt += 1
        if cum_sum > thrsh:
            break
    print(ratios)
    print("number of different grapheme clusters in {} = {}".format(language, len(ratios.keys())))
    print("{} grapheme clusters form {} of the text".format(cnt, thrsh))


def download_from_gcs(gcs_uri, dir):
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs://uri, got {gcs_uri}")
    _, _, bucket_and_path = gcs_uri.partition("://")
    bucket_name, _, blob_path = bucket_and_path.partition("/")
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, os.path.basename(blob_path))

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(filename)