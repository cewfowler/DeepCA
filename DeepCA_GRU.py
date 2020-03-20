import os;
import sys;

import json;
import numpy as np;
import tensorflow as tf;
import tensorflow.compat.v1.nn as tfv1;
from util.util import plot_stereo_spectrogram;

n_input = 3;
learning_rate = 0.001;
training_steps = 10000;
batch_size = 128;
n_hidden = 512;
n_output = 5;

"""
weights = {
    'out': tf.Variable(tf.zeros([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.zeros([n_output]))
}
"""


def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input]);

    # Generate a n_input-element sequence of inputs
    x = tf.split(x, n_input, 1);

    # 2-layer GRU with n_hidden units
    rnn_cell = tfv1.rnn_cell.MultiRNNCell([tfv1.rnn_cell.GRUCell(n_hidden), tfv1.rnn_cell.GRUCell(n_hidden)]);

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32);

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'];


def main():
    #plot_stereo_spectrogram(os.getcwd() + '/test.wav');
    print('Main');


if __name__ == '__main__':
    main();
