import numpy as np;
import tensorflow as tf;
from tensorflow.python.ops import gen_audio_ops as contrib_audio;
import librosa.feature;


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
