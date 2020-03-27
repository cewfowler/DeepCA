import matplotlib.pyplot as plt;
import numpy as np;
from scipy.io import wavfile;


def visualize_t_domain(file):
    sample_rate, data = wavfile.read(file);

    plt.figure(figsize=(10, 5));

    plt.title('Raw wave of ' + file);
    plt.xlabel('Time');
    plt.ylabel('Amplitude');

    plt.plot(np.linspace(0, len(data)/sample_rate, len(data)), data);
    plt.show();


def plot_stereo_spectrogram(wav_file):
    try:
        sample_rate, samples = wavfile.read(wav_file);
        print('Success reading wav file')
    except:
        print('Error reading wav file.');

    sample1 = [];
    sample2 = [];
    for vals in samples:
        sample1.append(vals[0]);
        sample2.append(vals[1]);

    plt.title("Spectrogram of " + wav_file)
    plt.subplot(221);
    plt.plot(sample1);
    plt.xlabel('Sample');
    plt.ylabel('Amplitude');

    plt.subplot(223);
    plt.specgram(sample1, Fs=sample_rate);
    plt.xlabel('Time');
    plt.ylabel('Frequency');

    plt.subplot(222);
    plt.plot(sample2);
    plt.xlabel('Sample');
    plt.ylabel('Amplitude');

    plt.subplot(224);
    plt.specgram(sample2, Fs=sample_rate);
    plt.xlabel('Time');
    plt.ylabel('Frequency');

    plt.show()
