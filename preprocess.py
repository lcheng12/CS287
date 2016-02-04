#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
<<<<<<< HEAD
import codecs

=======
import itertools
>>>>>>> f4a5d8728d28ff15d0d1d99088ba20c31d93ae88

def line_to_words(line, dataset):
    # Different preprocessing is used for these datasets.
    if dataset in ['SST1', 'SST2']:
        clean_line = clean_str_sst(line.strip())
    else:
        clean_line = clean_str(line.strip())
    words = clean_line.split(' ')
    words = words[1:]
    return words

def get_ngrams(words, n):
    return zip(*[words[i:] for i in range(n)])

# KW: returns a mapping strings to integers
def get_vocab(file_list, dataset='', ngram_limit=1):
    """
    Construct index feature dictionary.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    max_sent_len = 0
    word_to_idx = {}
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    words = line_to_words(line, dataset)
                    length = len(words)
                    max_sent_len = max(max_sent_len, ngram_limit * (2 * length - ngram_limit + 1) / 2)
                    for n in xrange(1, ngram_limit + 1):
                        grams = get_ngrams(words, n)
                        # print grams, n
                        for gram in grams:
                            if gram not in word_to_idx:
                                word_to_idx[gram] = idx
                                idx += 1
                    # print words
                    # print get_ngrams(words, 1)
                    # print get_ngrams(words, 2)
                    """
                    for word in words:
                        if word not in word_to_idx:
                            word_to_idx[word] = idx
                            idx += 1
                    """
    return max_sent_len, word_to_idx


def convert_data(data_name, word_to_idx, max_sent_len, dataset, ngram_limit=1, start_padding=0):
    """
    Convert data to padded word index features.
    EXTENSION: Change to allow for other word features, or bigrams.
    """
    # KW: ends up as a 2-D array
    features = []
    # KW: ends up as a 1-D array
    lbl = []
    with codecs.open(data_name, 'r', encoding="latin-1") as f:
        for line in f:
            words = line_to_words(line, dataset)
            # KW: y is the class
            y = int(line[0]) + 1
            sent = []
            # KW: All the ids of the words, dedupped
            for n in xrange(1, ngram_limit + 1):
                grams = get_ngrams(words, n)
                sent += [word_to_idx[gram] for gram in grams]
            #sent = [word_to_idx[word] for word in words]
            sent = list(set(sent))
            # end padding
            # KW: we add this to the end for uniform data
            if len(sent) < max_sent_len + start_padding:
                sent.extend([1] * (max_sent_len + start_padding - len(sent)))
            # start padding
            sent = [1]*start_padding + sent
            #print sent
            features.append(sent)
            
            lbl.append(y)
    return np.array(features, dtype=np.int32), np.array(lbl, dtype=np.int32)


# KW: Remove non-alphanumerics and excess whitespace
def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#KW: fix punctuation parentheses abbreviations, as well as removing non alphanumeric
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# Different data sets to try.
# Note: TREC has no development set.
# And SUBJ and MPQA have no splits (must use cross-validation)
FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                       "data/stsa.fine.dev",
                       "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.phrases.train",
                       "data/stsa.binary.dev",
                       "data/stsa.binary.test"),
              "TREC": ("data/TREC.train.all", None,
                       "data/TREC.test.all"),
              "SUBJ": ("data/subj.all", None, None),
              "MPQA": ("data/mpqa.all", None, None)}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test = FILE_PATHS[dataset]

    ngram_limit = 1
    
    # Features are just the words.
<<<<<<< HEAD
    max_sent_len, word_to_idx = get_vocab([train, valid, test], dataset)
=======
    max_sent_len, word_to_idx = get_vocab([train, valid, test], ngram_limit=ngram_limit)
>>>>>>> f4a5d8728d28ff15d0d1d99088ba20c31d93ae88

    # Dataset name
    train_input, train_output = convert_data(train, word_to_idx, max_sent_len,
                                             dataset, ngram_limit)

    # KW: if these sets exist
    if valid:
        valid_input, valid_output = convert_data(valid, word_to_idx, max_sent_len,
                                                 dataset, ngram_limit)
    if test:
        test_input, _ = convert_data(test, word_to_idx, max_sent_len,
                                     dataset, ngram_limit)

    V = len(word_to_idx) + 1
    print('Vocab size:', V)

    C = np.max(train_output)

    # KW: oh this takes care of dumping the matrix out for us! 
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
