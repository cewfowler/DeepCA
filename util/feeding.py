import numpy as np;
import tensorflow as tf;
from tensorflow.python.ops import gen_audio_ops as contrib_audio;
import librosa.feature;


def calculate_mfccs(samples, sr):
    mfcc = librosa.feature.mfcc(samples, sr=sr, n_mfcc=20);
    return mfcc;
