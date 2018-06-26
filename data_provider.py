# -*- coding: utf-8 -*-
import numpy as np
import random


def load_binary_file(file_name, dimension):
    features = np.fromfile(file_name, dtype=np.float32).reshape((-1, dimension))
    return features, features.shape[0]


class ListDataProvider(object):

    def __init__(self, x_file_list, y_file_list, n_ins=0, n_outs=0, shuffle=False):

        self.n_ins = n_ins
        self.n_outs = n_outs

        try:
            assert len(x_file_list) > 0
            assert len(y_file_list) > 0
            assert len(x_file_list) == len(y_file_list)
        except AssertionError:
            print 'x_file_list and y_file_list size exception'
            raise

        self.x_files_list = x_file_list
        self.y_files_list = y_file_list

        if shuffle:
            random.seed(271638)
            random.shuffle(self.x_files_list)
            random.seed(271638)
            random.shuffle(self.y_files_list)

        self.list_size = len(self.x_files_list)
        self.file_index = 0
        self.end_reading = False

    def __iter__(self):
        return self

    def reset(self):
        """
        When all the files in the file list have been used for DNN training,
        reset the data provider to start a new epoch.
        """
        self.file_index = 0
        self.end_reading = False

    def is_finish(self):
        return self.end_reading

    def load_one_partition(self):
        temp_set_x, temp_set_y = self.load_next_utterance()
        return temp_set_x, temp_set_y

    def load_next_utterance(self):

        temp_x = []
        temp_y = []

        batch_size = 4

        for i in xrange(batch_size):
            x_features, x_frame_number = load_binary_file(self.x_files_list[self.file_index], self.n_ins)
            y_features, y_frame_number = load_binary_file(self.y_files_list[self.file_index], self.n_outs)
            temp_x.append(x_features)
            temp_y.append(y_features)
            self.file_index += 1
            if self.file_index >= self.list_size:
                self.file_index = 0
                self.end_reading = True
                break

        return temp_x, temp_y

    def load_one_sample(self):

        x_features, x_frame_number = load_binary_file(self.x_files_list[self.file_index], self.n_ins)
        y_features, y_frame_number = load_binary_file(self.y_files_list[self.file_index], self.n_outs)

        self.file_index += 1
        if self.file_index >= self.list_size:
            self.file_index = 0
            self.end_reading = True

        return x_features, y_features








