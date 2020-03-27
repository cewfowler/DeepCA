# This model was created referencing DeepSpeech 1 and 2 (papers can be found at
# https://arxiv.org/abs/1412.5567 and https://arxiv.org/abs/1512.02595). The
# code can be found at https://github.com/mozilla/DeepSpeech.

import os;
import sys;

import json;
import numpy as np;
import tensorflow as tf;
import tensorflow.compat.v1.nn as tfv1;

n_input = 3;

relu_clip = 20;

# Number of connections at each layer
n_hidden = {2048, 2048, 2048, 512, 512};

batch_size = 128;
learning_rate = 0.001;

training_steps = 10000;
n_epochs = 100;

n_output = 5;

"""
weights = {
    'out': tf.Variable(tf.zeros([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.zeros([n_output]))
}
"""

# DeepSpeech implementation referenced
# Dense layers with relu activation
# h(t, l) = g( W(l) * h(t, l-1) + b(l) ), W = weights matrix, b = bias matrix,
#   g(z) = min{ max{0,z}, relu_clip } is clipped recitified-linear (ReLu)
#          activation function
def dense(name, x, units, dropout_rate=None):
    with tfv1.variable_scope(name):
        bias = tfv1.get_variable(name='bias', \
                                shape=[units], \
                                initializer=tfv1.zeros_initializer());
        weights = tfv1.get_variable(name='weights', \
                                    shape=[x.shape[-1], units], \
                                    initializer=tfv1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"));

        output = tf.nn.bias_add(tf.matmul(x, weights), bias);

        output = tf.minimum(tf.nn.relu(output), relu_clip);


# GRU layers
def gru_impl(name, x, units, seq_length, prev_state, reuse):
    with tfv1.variable_scope(name):
        # GRU cell with n_hidden units
        gru_cell = tfv1.rnn_cell.GRUCell(units, \
                                        forget_bias=0, \
                                        reuse=reuse)

    # generate prediction
    outputs, states = gru_cell(inputs=x, \
                                dtype=tf.float32, \
                                sequence_length=seq_length, \
                                initial_state=prev_state);

    return output, output_state;

def create_model(batch_x, seq_length, dropout, reuse=False, batch_size=None, prev_state=None):
    layers = {};

    # First 3 layers (Dense with ReLu activation)
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, n_hidden[0], dropout_rate=dropout[0]);
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, n_hidden[1], dropout_rate=dropout[1]);
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, n_hidden[2], dropout_rate=dropout[2]);

    layer_3 = tf.reshape(layer_3, [-1, batch_size, n_hidden[3]])

    # Next 2 layers (GRU)
    output_1, out_state_1 = gru_impl('layer_4', layer_3, n_hidden[3], seq_length, prev_state[0], reuse);
    layers['layer_4'] = output_1;
    layers['gru_out_state_1'] = out_state_1;

    output_2, out_state_2 = gru_impl('layer_5', output_1, n_hidden[4], seq_length, prev_state[1], reuse);
    layers['layer_5'] = output_2;
    layers['gru_out_state_2'] = out_state_2;


def main():
    print('Main');


if __name__ == '__main__':
    main();
