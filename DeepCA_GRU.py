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
n_input = 256;
n_hidden = [2048, 1024, 512, 256, 256, 512];
n_output = 256;

n_epochs = 100;


def create_model():
    """
    tf.keras.backend.clear_session();
    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    session = tf.Session(config=config_proto)
    set_session(session)
    """

    trainX = np.ones((2048, 360, 3));
    trainY = np.ones((2048, 256));
    # to_categorical -> converts class vector to binary class matrix, for use
    # with categorical_crossentropy loss function
    #trainY = to_categorical(trainY, num_classes=n_output);

    model = Sequential();

    # Add dense layers (first 3 layers)
    # TODO: figure out input dimensions to model
    model.add(Dense(n_hidden[0], input_shape=(batch_size, n_input), activation='relu'));
    model.add(Dense(n_hidden[1], activation='relu'));
    model.add(Dense(n_hidden[2], activation='relu'));

    # TODO: figure out how to properly reshape
    model.add(Reshape((-1, n_hidden[3])));

    # Add recurrent GRU layers (layers 4 and 5)
    model.add(GRU(n_hidden[3], \
                  activation='tanh', \
                  recurrent_activation='sigmoid', \
                  return_sequences=True));
    #model.add(Reshape((-1, n_hidden[4])));
    model.add(GRU(n_hidden[4], \
                  activation='tanh', \
                  recurrent_activation='sigmoid'));

    # Add another dense layer
    model.add(Dense(n_hidden[5], activation='relu'));

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

    # Fit model to input, output
    history = model.fit(trainX, trainY, epochs=n_epochs)

    # Save model weights
    model.save_weights("model.h5");


def main():
    print('Main');


if __name__ == '__main__':
    main();
