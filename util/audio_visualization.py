import matplotlib.pyplot as plt;
import numpy as np;
import librosa;
import librosa.display as display
from .feeding import calculate_mfccs


# Visualize time domain of input audio file
def visualize_t_domain(file):
    data, sr = wavfile.read(file);

    plt.figure(figsize=(10, 5));

    plt.title('Raw wave of ' + file);
    plt.xlabel('Time');
    plt.ylabel('Amplitude');

    plt.plot(np.linspace(0, len(data)/sr, len(data)), data);
    plt.show();


# Plot spectrogram of input audio file
def plot_spectrogram(wav_file):
    try:
        data, sr =librosa.load(wav_file);
        print('Success reading wav file')
    except:
        print('Error reading wav file.');

    # Get short term Fourier transform and amplitudes
    X = librosa.stft(data);
    Xdb = librosa.amplitude_to_db(abs(X));

    # Plot spectrogram
    plt.figure(figsize=(14,5));
    display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz');
    plt.colorbar();
    plt.show();


def plot_mfccs(wav_file):
    data, sr = librosa.load(wav_file);
    mfcc = calculate_mfccs(data, sr);

    plt.figure(figsize=(14,5));
    display.specshow(mfcc, x_axis='time');
    plt.colorbar();
    plt.title('MFCCs for ' + wav_file);
    plt.tight_layout();
    plt.show();
