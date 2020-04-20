#----------------------------------------------------------------------------------------
#CNN Model for use Wave Analysis for STT#
#Created on theoretical understanding of CNNs and Audio Processing for DeepSpeech#
#Along with several KERAS/TF Documentation and Tutorials#
#----------------------------------------------------------------------------------------

#Importing the basic libraries for use in model and data preprocessing
import os
import sys
import numpy as np
import pandas as pd
from random                  import randint
from sklearn                 import metrics
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from DeepCA_AudioAnalyzer    import AudioAnalyzer

#Importing CNN model specific tools from the KERAS library
#along with tensorflow itself
import tensorflow as tf
from tensorflow.keras            import backend as K
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Activation, Dense, Input, Dropout, Flatten
from tensorflow.keras.layers     import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics    import categorical_crossentropy
from tensorflow.keras.utils      import plot_model, to_categorical #np_utils ???

#Model Parameters
num_rows      = 40
num_columns   = 174
num_channels  = 1
kernel_size   = 2
pool_size     = 2
batch_size    = 256
epoch_size    = 72 #size for maximum accuracy
filter_size   = 16
learning_rate = 0.001
#----------------------#

#Constructor
#--------------------------------#
if __name__ == '__main__':
    main()
#--------------------------------#

#Function to create our model on 3 parameters, a training set, a data set, and labels
def create(num_labels):
    
    #Creating Sequential Keras CNN
    model = tf.keras.Sequential()

    #Constructing our CNN model using 2-dimensional convolutional layers, with 1 input, 3 deep layers, and 1 output layer
    model.add(Conv2D(filters = filter_size, kernel_size = kernel_size, input_shape = (num_rows, num_columns, num_channels), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0,2))

    model.add(Conv2D(filters = filter_size*2, kernel_size = kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0,2))

    model.add(Conv2D(filters = filter_size*4, kernel_size = kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0,2))

    model.add(Conv2D(filters = filter_size*8, kernel_size = kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Dense(num_labels, activation = 'softmax'))

    #Flatten based on parameter, or simply ???
    #model.add(Flatten(input_shape = (28,28)))

    #Get the model summary for current model, pre-training
    model.summary()

    #compiling the model using ADAM as optimizer, above learning rate of 0.001, and Sparse Categorical Crossentropy
    model.compile(Adam(lr = learning_rate), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    #save the data to CSV file
    model.to_csv('model_output.csv')

    #Prints the model object
    plot_model(model, show_layer_names = True, show_shapes = True, to_file='model_output.png')

    return model

#Trains the model on function call
def train(model, data, labels, batch, epochs):
    #with open("model_output.csv", "r") as csv:
    #    model = model.from_csv(csv.read())

    batch = batch_size  #global batch
    epochs = epoch_size #global epoch

    fitted = model.fit(data, labels, batch, epochs,shuffle = True, verbose = 2) #Fits and trains the imported model
    
    return fitted

def main():

    #Extract audio samples and create CSV
    AudioAnalyzer.audio_transform()

    # Set the path to the dataset, and turn it to .CSV file
    fulldatasetpath = 'C:/Users/Vitaliy/Desktop/Desktop Folder/Engineering/Senior Design/Sample Dataset/'
    metadata = pd.read_csv(fulldatasetpath + 'dataset.csv')

    #Feature extraction list
    features = []

    #Opens testing set text file
    txt_file = open("C:/Users/Vitaliy/Desktop/Desktop Folder/Engineering/Senior Design/Sample Dataset/testing_list.txt", "r")
        
    #Transforms data, and adds it to our array
    for file_path in txt_file:

        file_name = os.path.join(fulldatasetpath, file_path)
        file_name = file_name.strip()
        
        #Pulls the class label off the file
        class_label = file_name.split('/')[-2]
        data = AudioAnalyzer.feature_extraction(file_name)
        
        #Appends our extracted features to above array
        features.append([data, class_label])

    #Create Data Frame, and 2 numpy arrays to store data
    feature_dataframe = pd.DataFrame(features, columns = ['Feature', 'Class_Label'])
    feature_set = np.array(feature_dataframe.Feature.to_list())
    label_set   = np.array(feature_dataframe.Class_Label.to_list())

    #Label encoding for the CNN model
    label_enc     = LabelEncoder()
    enc_label_set = label_enc.fit_transform(label_set)

    #Split our data for training
    training_set, data_set, training_label_set, label_set = train_test_split(feature_set, enc_label_set, test_size = 0.2, random_state = 64)

    #Reshaping the labels, training and data sets
    training_set = training_set.reshape(training_set.shape[0], num_rows, num_columns, num_channels)
    data_set     = data_set.reshape(data_set.shape[0], num_rows, num_columns, num_channels)
    num_labels   = enc_label_set.shape[1]

    #Create and train our model
    model        = create(enc_label_set)
    fitted_model = train(training_set, training_label_set, batch_size, epoch_size)

    # Evaluating the model on the training and testing set
    score = fitted_model.evaluate(training_set, training_label_set, verbose = 2)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(data_set, label_set, verbose = 2)
    print("Testing Accuracy: ", score[1])

    #save the new model data to CSV file
    fitted_model.to_csv('fitted_model_output.csv')