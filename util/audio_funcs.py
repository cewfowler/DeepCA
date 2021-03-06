import wave
import tensorflow as tf
import numpy as np
import io
import os
import librosa.feature
import librosa

# Default audio settings
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)

# Calculate the mfccs for input
def calculate_mfccs(wavfile):
    samples, sample_rate = librosa.load(wavfile)
    mfcc = librosa.feature.mfcc(samples, sr=sample_rate, n_mfcc=20, dct_type=2);
    return mfcc;

# Read the frames from wav file. Input frame duration.
def read_frames(wav_file, frame_duration_ms):
    file = wave.open(wav_file, 'r')
    frame_size = int(DEFAULT_RATE * (frame_duration_ms/1000))
    while True:
        try:
            data = file.readframes(frame_size)
            # If the PCM is shorter than the frame duration then skip the data. Removable
            if( ((len(data) // (DEFAULT_CHANNELS * DEFAULT_WIDTH)) / DEFAULT_RATE) * 1000 < frame_duration_ms ):
                break
            yield data
        except EOFError:
            wave.close(wav_file)
            break

# Create batch set. Make sure output_types = padded_shapes in terms of arguments.
def create_batch_set(wav_file, frame_duration_ms, batch_size):
    dataset = tf.data.Dataset.from_generator(read_frames, output_types=(tf.int32, tf.int32, tf.int32, tf.int32))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([], [], [], []))
    return dataset
