# This model was created referencing DeepSpeech 1 and 2 (papers can be found at
# https://arxiv.org/abs/1412.5567 and https://arxiv.org/abs/1512.02595). The
# code can be found at https://github.com/mozilla/DeepSpeech.

import os;
import sys;
import json;
import numpy as np;
import tensorflow as tf;

from keras.models import Sequential, model_from_json;
from keras.layers import Dense, Flatten, GRU, Reshape, Bidirectional, Activation;
from keras.layers.normalization import BatchNormalization;
#from keras.utils import to_categorical;

n_input = 3;
learning_rate = 0.001;

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
    # Input must be described as input shape to work with recurrent layers
    # TODO: figure out input dimensions to model, DeepSpeech uses spectrogram as input
    model.add(Dense(n_hidden[0], input_shape=(n_input,1), use_bias=False));
    model.add(BatchNormalization());
    model.add(Activation("relu"));

    model.add(Dense(n_hidden[1], use_bias=False));
    model.add(BatchNormalization());
    model.add(Activation("relu"));

    model.add(Dense(n_hidden[2], use_bias=False));
    model.add(BatchNormalization());
    model.add(Activation("relu"));

    # 3 dimensions of inputs to RNNs (LSTM only?):
    # 1. batch size, 2. time steps, 3. input dimension
    # https://www.researchgate.net/post/What_are_the_input_output_dimensions_when_training_a_simple_Recurrent_or_LSTM_neural_network
    # Add recurrent GRU layers (layers 4 and 5)
    model.add(Bidirectional(GRU(n_hidden[3], \
                                activation='tanh', \
                                recurrent_activation='sigmoid', \
                                return_sequences=True)));
    # TODO: need to add input_length? See keras docs
    # https://keras.io/layers/recurrent/
    model.add(Bidirectional(GRU(n_hidden[4], \
                                activation='tanh', \
                                recurrent_activation='sigmoid',
                                return_sequences=True)));

    # TODO: add conv layer?

    # Flatten the output from GRU layers
    model.add(Flatten());

    # Add output layer
    model.add(Dense(n_output, use_bias=False));
    model.add(BatchNormalization());
    model.add(Activation("softmax"));

    # Compile model
    # TODO: categorical_crossentropy vs sparse_categorical_crossentropy
    model.compile(loss='sparse_categorical_crossentropy', \
                  optimizer='adam', \
                  metrics=['accuracy']);

    # Save model as json file
    model_json = model.to_json();
    with open("model.json", "w") as json_file:
        json_file.write(model_json);

    model.summary();


def train_model(inputs, labels, num_epochs):
    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read());

    history = model.fit(inputs, labels, epochs=num_epochs);

    model.save_weights("model.h5");

    return history;


def main():
    create_model();
    print('Main');


if __name__ == '__main__':
    main();
