# This model was created referencing DeepSpeech 1 and 2 (papers can be found at
# https://arxiv.org/abs/1412.5567 and https://arxiv.org/abs/1512.02595). The
# code can be found at https://github.com/mozilla/DeepSpeech.

import os;
import sys;
import json;
import numpy as np;
import tensorflow as tf;
import pandas as pd;

from keras.models import Sequential, model_from_json;
from keras.layers import Dense, Flatten, GRU, Reshape, Bidirectional, Activation;
from keras.layers.normalization import BatchNormalization;
from keras.utils import plot_model, to_categorical;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from util.feeding import getCSVFeatures;
from util.visualize_model import visualize_accuracy;

n_input = 3;
learning_rate = 0.001;

relu_clip = 20;

# Number of connections at each layer
n_input = 8000;
n_hidden = [2048, 1024, 512, 128, 128];
n_output = 256;

batch_size = 256;
n_epochs = 72;

frame_duration_ms = 20;


def create_model():
    model = Sequential();

    # Add dense layers (first 3 layers)
    # Input must be described as input shape to work with recurrent layers
    # TODO: figure out input dimensions to model, DeepSpeech uses spectrogram as input
    model.add(Dense(n_hidden[0], input_shape=(), use_bias=False));
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
    plot_model(model, show_layer_names = True, show_shapes = True, to_file='model_output.png')

    return model;


def train_model(inputs, labels, num_epochs=n_epochs):
    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read());

    history = model.fit(inputs, labels, batch_size, epochs=num_epochs, shuffle=True, verbose=2);

    model.save_weights("model.h5");

    return history;


def main():
    features = getCSVFeatures('./spliced_audio', './csv');

    #Create Data Frame, and 2 numpy arrays to store data
    feature_dataframe = pd.DataFrame(features, columns = ['Feature', 'Class_Label']);
    feature_set = np.array(feature_dataframe.Feature.to_list());
    label_set   = np.array(feature_dataframe.Class_Label.to_list());

    #Label encoding for the CNN model
    label_enc     = LabelEncoder();
    enc_label_set = label_enc.fit_transform(label_set);

    #Split our data for training
    training_set, data_set, training_label_set, label_set = train_test_split(feature_set, enc_label_set, test_size = 0.2, random_state = 64);

    #Reshaping the labels, training and data sets
    training_set = training_set.reshape(training_set.shape[0], num_rows, num_columns, num_channels);
    data_set     = data_set.reshape(data_set.shape[0], num_rows, num_columns, num_channels);
    num_labels   = enc_label_set.shape[1];

    create_model();
    hist = train(training_set, training_label_set);

    # Evaluating the model on the training and testing set
    visualize_accuracy(hist);

    hist.to_csv('fitted_model_output.csv');


if __name__ == '__main__':
    main();
