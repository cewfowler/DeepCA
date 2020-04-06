# This model was created referencing DeepSpeech 1 and 2 (papers can be found at
# https://arxiv.org/abs/1412.5567 and https://arxiv.org/abs/1512.02595). The
# code can be found at https://github.com/mozilla/DeepSpeech.

import os;
import sys;
import json;
import numpy as np;
import tensorflow as tf;

from keras.models import Sequential;
from keras.layers import Dense, Flatten, GRU, Reshape, Bidirectional;
#from keras.utils import to_categorical;

"""
import tensorflow.compat.v1 as tf;

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1.keras.backend import set_session
"""

n_input = 3;

relu_clip = 20;

# Number of connections at each layer
batch_size = 64;
n_input = 8000;
n_hidden = [2048, 1024, 512, 128, 128];
n_output = 256;

n_epochs = 1;


def create_model():

    model = Sequential();

    # Add dense layers (first 3 layers)
    # TODO: figure out input dimensions to model
    model.add(Dense(n_hidden[0], input_shape=(n_input, batch_size), activation='relu'));
    model.add(Dense(n_hidden[1], activation='relu'));
    model.add(Dense(n_hidden[2], activation='relu'));

    # Add recurrent GRU layers (layers 4 and 5)
    model.add(Bidirectional(GRU(n_hidden[3], \
                                activation='tanh', \
                                recurrent_activation='sigmoid', \
                                return_sequences=True)));
    model.add(Bidirectional(GRU(n_hidden[4], \
                                activation='tanh', \
                                recurrent_activation='sigmoid',
                                return_sequences=True)));

    # Flatten the output from GRU layers
    model.add(Flatten());

    # Add output layer
    model.add(Dense(n_output, activation='softmax'));

    # Compile model
    # TODO: use categorical_crossentropy or sparse_categorical_crossentropy
    # sparse_categorical_crossentropy does not require one hot encoding, ie.
    # does not require to_categorical and saves significant memory if there
    # is a large number of categories
    model.compile(loss='sparse_categorical_crossentropy', \
                  optimizer='adam', \
                  metrics=['accuracy']);

    # Save model as json file
    model_json = model.to_json();
    with open("model.json", "w") as json_file:
        json_file.write(model_json);

    model.summary();


def main():
    create_model();
    print('Main');


if __name__ == '__main__':
    main();
