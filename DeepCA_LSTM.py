import os
import sys

import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 3;
learning_rate = 0.001;
training_steps = 10000;
batch_size = 128;
n_hidden = 512;
n_output = 5;

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]));
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}


def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input]);

    # Generate a n_input-element sequence of inputs
    x = tf.split(x, n_input, 1);

    # 2-layer LSTM with n_hidden units
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)]);

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32);

    # there are n_input outputs but we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'];


def main():
    print("Beginning...");


if __name__ == '__main__':
    main();
