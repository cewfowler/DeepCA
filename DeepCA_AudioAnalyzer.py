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
    def audio_transform(file_path):

        #wav data array with file path assignment
        wav_data = []
        file_name = file_path

        #Creates instance of wavfilehelper
        wavfilehelper = WavFileHelper()

        #Transforms data, and adds it to our array
        data = wavfilehelper.read_file_properties(file_name)
        wav_data.append()

        return wav_data

    # Convert into a Panda dataframe
    def convert_data(wav_data):
        data_frame = pd.DataFrame(wav_data, columns=['num_channels','sample_rate','bit_depth'])

        return data_frame

    #Extracts and returns the scaled MFCCs of the .WAV file
    def feature_extraction(file_name):

        #Exception handling?
        #Takes the file, and extracts the sampling rate at 22.05 kHZ
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 20)

        #Scale the MFCC
        scaled_mfccs = np.mean(mfccs.T, axis = 0)

        return scaled_mfccs

    #-------------------------------------Reference for pulling file from OS path---------------------------------------#
    #file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    #-------------------------------------------------------------------------------------------------------------------#
