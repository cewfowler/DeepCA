# This model was created referencing DeepSpeech 1 and 2 (papers can be found at
# https://arxiv.org/abs/1412.5567 and https://arxiv.org/abs/1512.02595). The
# code can be found at https://github.com/mozilla/DeepSpeech.

import os;
import sys;
import json;
import numpy as np;
import tensorflow as tf;

from keras.models import Sequential;
from keras.layers import Dense, GRU, Reshape;

"""
import tensorflow.compat.v1 as tf;

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1.keras.backend import set_session
"""

n_input = 3;

relu_clip = 20;

# Number of connections at each layer
n_hidden = [2048, 2048, 2048, 512, 512, 2048, 256];

batch_size = 128;
learning_rate = 0.001;

training_steps = 10000;
n_epochs = 100;

n_output = 5;


def create_model():
    """
    tf.keras.backend.clear_session();
    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    session = tf.Session(config=config_proto)
    set_session(session)
    """

    x = np.ones((2048, 360, 3));
    y = np.ones((2048, 256));

    model = Sequential();

    # Add dense layers (first 3 layers)
    model.add(Dense(n_hidden[0], activation='relu'));
    model.add(Dense(n_hidden[1], activation='relu'));
    model.add(Dense(n_hidden[2], activation='relu'));

    model.add(Reshape((-1, n_hidden[3])));

    # Add recurrent GRU layers (layers 4 and 5)
    model.add(GRU(n_hidden[3], activation='tanh', recurrent_activation='sigmoid'));
    model.add(Reshape((-1, n_hidden[4])));

    model.add(GRU(n_hidden[4], activation='tanh', recurrent_activation='sigmoid'));

    # Add another dense layer
    model.add(Dense(n_hidden[5], input_dim=n_hidden[4], activation='relu'));

    # Add output layer
    model.add(Dense(n_hidden[6], input_dim=n_hidden[5], activation='softmax'));

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);

    model_json = model.to_json();
    with open("model.json", "w") as json_file:
        json_file.write(model_json);
    model.fit(x,y)

    model.save_weights("model.h5");


def main():
    print('Main');


if __name__ == '__main__':
    main();
