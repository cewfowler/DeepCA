#----------------------------------------------------------------------------------------
#CNN Model for use Wave Analysis for STT#
#Created on theoretical understanding of CNNs and Audio Processing for DeepSpeech#
#Along with several KERAS/TF Documentation and Tutorials#
#----------------------------------------------------------------------------------------

#Importing the basic libraries for use in model and data preprocessing
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

#Importing CNN model specific tools from the KERAS library
#along with tensorflow itself
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import plot_model

#Model Parameters
batch_size    = 32
epoch_size    = 10 #size for maximum accuracy
learning_rate = 0.001

def create():
    #Creating Sequential Keras CNN
    model = tf.keras.Sequential([

        #Input parameter 1, dense input layer of taking 1 dimension, 16 perceptrons, RELU activation function
        Dense(64, input_shape = (1,), activation = 'relu'),

        #The rest of the deep layers
        Dense(128, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(32, activation = 'relu'),

        #Output function parameter using softmax function
        Dense(16, activation = 'softmax')
    ])

    #Flatten based on parameter, or simply ???
    #model.add(Flatten(input_shape = (28,28)))

    #Get the model summary for current model, pre-training
    model.summary()

    #compiling the model using ADAM as optimizer, above learning rate of 0.001, and Sparse Categorical Crossentropy
    model.compile(Adam(lr=.0001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    #save the data to CSV file
    model.to_csv('model_output.csv')

    #Prints the model object
    plot_model(model, show_layer_names = True, show_shapes = True, to_file='model_output.png')

#Trains the model on function call
def train(data, labels, batch, epochs):
    with open("model_output.csv", "r") as csv:
        model = model.from_csv(csv.read())

    batch = batch_size  #global batch
    epochs = epoch_size #global epoch

    fitted = model.fit(data, labels, batch, epochs,shuffle = True, verbose = 2) #Fits and trains the imported model
    
    return fitted