import numpy as np;
import tensorflow as tf;
from tensorflow.python.ops import gen_audio_ops as contrib_audio;
import librosa.feature;
import audio_funcs import read_frames


def calculate_mfccs(samples, sr):
    mfcc = librosa.feature.mfcc(samples, sr=sr, n_mfcc=20);
    return mfcc;


# Ensure that same char file is used for all labels
def encodeLabels(label_file_path, chars_file_path='../language/chars.txt'):
    chars = [];
    res = [];

    # Extract all the CA chars from the chars file
    with open(chars_file_path, 'r') as c:
        for line in c:
            if line[0] == '#':
                continue;

            chars.append(line[:-1]);

    with open(label_file_path, 'r') as l:
        for line in l:
            for char in line:
                try:
                    res.append(chars.index(char));
                except:
                    print("Unable to find char (" + char + "), exiting...");
                    return -1;

    return res;


# Get features from CSV and audio file
def getCSVFeatures(audio_file, csv_file):
    meta = pd.read_csv(csv_file);
    features = [];

    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():

        #Joins the each file name in a file_name tuple
        file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))

        #Name labels and extract features of .WAV data with the feature extraction tool
        class_label = row["class_name"]
        data = AudioAnalyzer.feature_extraction(file_name)

        #Appends our extracted features to above array
        features.append([data, class_label]);

    return features;
