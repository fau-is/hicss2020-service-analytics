from __future__ import division
import csv
import copy
import numpy as np

try:
    from itertools import izip as zip
except ImportError:
    pass
import util
from sklearn.model_selection import StratifiedKFold, KFold


class Preprocess_Manager(object):
    num_folds = 1
    data_dir = ""
    ascii_offset = 161

    date_format = "%d.%m.%Y-%H:%M:%S"

    # meta structures
    train_index_per_fold = list()
    test_index_per_fold = list()
    iteration_cross_validation = 0

    chars = list()
    char_indices = dict()
    indices_char = dict()
    target_char_indices = dict()
    target_indices_char = dict()
    elems_per_fold = 0

    lines = list()
    features_additional_sequences = list()

    # number of attributes
    num_features_all = 0
    num_features_activities = 0
    max_sequence_length = 0
    num_features_additional = 0
    num_attributes_standard = 3  # case, event, timestamp

    @classmethod
    def __init__(self, args):
        self.num_folds = args.num_folds
        self.data_dir = args.data_dir + args.data_set

        util.llprint("Create structures...")
        lines = []
        caseids = []
        lastcase = ''
        line = ''
        firstLine = True
        numlines = 0
        check_additional_features = True

        features_additional_attributes = []
        features_additional_events = []
        features_additional_sequences = []

        csvfile = open(self.data_dir, 'r')
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        next(spamreader, None)
        for row in spamreader:

            # initial setting of additional features
            if check_additional_features:
                if len(row) == self.num_attributes_standard:
                    util.llprint("No additional attributes.\n")
                else:
                    self.num_features_additional = len(row) - self.num_attributes_standard
                    util.llprint("Number of additional attributes: %d\n" % self.num_features_additional)
                check_additional_features = False

            if row[0] != lastcase:
                caseids.append(row[0])
                lastcase = row[0]
                if not firstLine:
                    lines.append(line)
                    if self.num_features_additional > 0:
                        features_additional_sequences.append(features_additional_events)
                line = ''
                if self.num_features_additional > 0:
                    features_additional_events = []
                numlines += 1

            # get values of additional attributes
            if self.num_features_additional > 0:
                for index in range(self.num_attributes_standard,
                                   self.num_attributes_standard + self.num_features_additional):
                    features_additional_attributes.append(row[index])
                features_additional_events.append(features_additional_attributes)
                features_additional_attributes = []

            # add activity to a case
            line += chr(int(row[1]) + self.ascii_offset)
            firstLine = False

        lines.append(line)
        if self.num_features_additional > 0:
            features_additional_sequences.append(features_additional_events)
        numlines += 1

        # get elements per fold in case of split evaluation
        util.llprint("Loading Data starts... \n")
        self.elems_per_fold = int(round(numlines / self.num_folds))

        util.llprint("Calc max length of sequence\n")
        lines = list(map(lambda x: x + '!', lines))
        self.max_sequence_length = max(map(lambda x: len(x), lines))
        util.llprint("Max length of sequence: %d\n" % self.max_sequence_length)

        util.llprint("Beginn calculation of total chars and total target chars... \n")
        self.chars = list(map(lambda x: set(x), lines))
        self.chars = list(set().union(*self.chars))
        self.chars.sort()
        self.target_chars = copy.copy(self.chars)
        self.chars.remove('!')
        util.llprint("Total chars: %d, target chars: %d\n" % (len(self.chars), len(self.target_chars)))

        util.llprint("Beginn creation of dicts for char handling... \n")
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.target_char_indices = dict((c, i) for i, c in enumerate(self.target_chars))
        self.target_indices_char = dict((i, c) for i, c in enumerate(self.target_chars))
        util.llprint("Dics for char handling created\n")

        # set feature variables
        self.num_features_activities = len(self.chars)
        self.num_features_all = self.num_features_activities + self.num_features_additional

        # set structure variables
        self.lines = lines
        self.caseids = caseids
        if self.num_features_additional > 0:
            self.features_additional_sequences = features_additional_sequences

        # init validation
        kFold = KFold(n_splits=self.num_folds, random_state=0, shuffle=True)

        for train_index, test_index in kFold.split(lines):
            self.train_index_per_fold.append(train_index)
            self.test_index_per_fold.append(test_index)

    @classmethod
    def create_and_encode_training_set(self, args):

        lines = list()
        lines_add = list()

        for index, value in enumerate(self.train_index_per_fold[self.iteration_cross_validation]):
            lines.append(self.lines[value])
            if self.num_features_additional > 0:
                lines_add.append(self.features_additional_sequences[value])

        step = 1
        sentences = []
        softness = 0
        next_chars = []
        if self.num_features_additional > 0:
            sentences_add = []

        if self.num_features_additional > 0:
            for line, line_add in zip(lines, lines_add):
                for i in range(0, len(line), step):
                    if i == 0:
                        continue
                    sentences.append(line[0: i])
                    sentences_add.append(line_add[0:i])
                    next_chars.append(line[i])

            util.llprint("\nnb sequences: %d" % len(sentences))
            util.llprint("\nadd sequences: %d" % len(sentences_add))
        else:
            for t, line in enumerate(lines):
                for i in range(0, len(line), step):
                    if i == 0:
                        continue
                    sentences.append(line[0: i])
                    next_chars.append(line[i])

            util.llprint("\nnb sequences: %d" % len(sentences))

        util.llprint("\nnb Vectorization ...")

        X = np.zeros((len(sentences), self.max_sequence_length, self.num_features_all), dtype=np.float64)
        Y = np.zeros((len(sentences), len(self.target_chars)), dtype=np.float64)

        for i, sentence in enumerate(sentences):

            leftpad = self.max_sequence_length - len(sentence)
            if self.num_features_additional > 0:
                sentence_add = sentences_add[i]

            # set training set data
            for t, char in enumerate(sentence):
                for c in self.chars:
                    if c == char:
                        X[i, t + leftpad, self.char_indices[c]] = 1.0

                # add additional attributes
                if self.num_features_all > 0:
                    for x in range(0, self.num_features_additional):
                        X[i, t + leftpad, len(self.chars) + x] = sentence_add[t][x]

            # set training set label
            for c in self.target_chars:
                if c == next_chars[i]:
                    Y[i, self.target_char_indices[c]] = 1 - softness
                else:
                    Y[i, self.target_char_indices[c]] = softness / (len(self.target_chars) - 1)

        num_features_all = self.num_features_all
        num_features_activities = self.num_features_activities

        return X, Y, self.max_sequence_length, num_features_all, num_features_activities

    @classmethod
    def create_test_set(self):
        lines = list()
        caseids = list()
        lines_add = list()

        for index, value in enumerate(self.test_index_per_fold[self.iteration_cross_validation]):
            lines.append(self.lines[value])
            caseids.append(self.caseids[value])
            if self.num_features_additional > 0:
                lines_add.append(self.features_additional_sequences[value])

        if self.num_features_additional > 0:
            return lines, caseids, lines_add, self.max_sequence_length, self.num_features_all, self.num_features_activities
        else:
            return lines, caseids, self.max_sequence_length, self.num_features_all, self.num_features_activities

    @classmethod
    def encode_test_set(self, sentence, batch_size):

        X = np.zeros((batch_size, self.max_sequence_length, self.num_features_all), dtype=np.float32)
        leftpad = self.max_sequence_length - len(sentence)

        for t, char in enumerate(sentence):
            for c in self.chars:
                if c == char:
                    X[0, t + leftpad, self.char_indices[c]] = 1.0
        return X

    @classmethod
    def encode_test_set_add(self, args, sentence, sentence_add, batch_size):

        X = np.zeros((1, self.max_sequence_length, self.num_features_all), dtype=np.float32)
        leftpad = self.max_sequence_length - len(sentence)

        for t, char in enumerate(sentence):
            for c in self.chars:
                if c == char:
                    X[0, t + leftpad, self.char_indices[c]] = 1.0

            for x in range(0, self.num_features_additional):
                X[batch_size - 1, t + leftpad, len(self.chars) + x] = sentence_add[t][x]

        num_features_all = self.num_features_all
        num_features_activities = self.num_features_activities

        return X, num_features_all, num_features_activities

    @classmethod
    def getSymbol(self, predictions):
        maxPrediction = 0
        symbol = ''
        i = 0;
        for prediction in predictions:
            if prediction >= maxPrediction:
                maxPrediction = prediction
                symbol = self.target_indices_char[i]
            i += 1
        return symbol
