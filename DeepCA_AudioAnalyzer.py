#Python audio-data classifier/processor
#Written through the assistance of research papers and tutorials:
#https://arxiv.org/abs/1709.01922 -- Research Paper
#https://medium.com/@ageitgey/machine-learning-is-fun-part-6-how-to-do-speech-recognition-with-deep-learning-28293c162f7a -- Ideas
#https://gist.github.com/mikesmales/67fc6ec36078a94bb62f0f455d4d6a38 -- .WAV file helper function
#---------------------------------------------------------------------------------
#This processing file/library takes sound data in .WAV and performs
#several known analyses, transforms, and feature extraction in order
#to prepare the data for being trained/learned in the machine learning model
#---------------------------------------------------------------------------------

#Imports major libraries for use in functions
import os
import sys
import librosa
import numpy as np
import pandas as pd

#Import the WAV file helper
from wavfilehelper import WavFileHelper

#Creates our audio analyzer class
class AudioAnalyzer():
    
    #Discrete data transformation using WavFileHelper
    def audio_transform(self):

        #Base path to file
        base_path = 'C:/Users/Vitaliy/Desktop/Desktop Folder/Engineering/Senior Design/Sample Dataset/'
        
        #wav data array with file path assignment
        wav_data = []

        #Creates instance of wavfilehelper
        wavfilehelper = WavFileHelper()

        #Opens testing set text file
        txt_file = open("C:/Users/Vitaliy/Desktop/Desktop Folder/Engineering/Senior Design/Sample Dataset/testing_list.txt", "r")
        
        #Transforms data, and adds it to our array
        for file_path in txt_file:

            file_name = os.path.join(base_path, file_path)
            file_name = file_name.strip()
            data = wavfilehelper.read_file_properties(file_name)
            wav_data.append(data)

        # Convert into a Panda dataframe and to csv
        wav_df = pd.DataFrame(wav_data, columns=['num_channels','sample_rate','bit_depth','class_label'])
        wav_df.to_csv('dataset.csv')

        return wav_df

    #Extracts and returns the scaled MFCCs of the .WAV file
    def feature_extraction(self, file_name):

        #Exception handling?
        #Takes the file, and extracts the sampling rate at 22.05 kHZ
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)

        #Scale the MFCC
        scaled_mfccs = np.mean(mfccs.T, axis = 0)

        return scaled_mfccs