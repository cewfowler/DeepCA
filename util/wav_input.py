import sys
import os
import tensorflow as tf
from multiprocessing import Process, cpu_count
from audio import DEFAULT_FORMAT, AudioFile, split_audio_file


def transcribe_file(audio_path):
    # split_audio_file parameter variable settings
    batchsize = 128
    # Aggressiveness has to be within 0-3 for VAD split to work.
    aggressive = 1
    outlierdurms = 20
    outlierbatchsize = 64

    # from .... import  DS model

    # Makes sure there are cores present in a system to be used
    try:
        num_processes = cpu_count()

    except NotImplementedError:
        num_processes = 1

    with AudioFile(audio_path, as_path=True) as wav_path:
        dataset = split_audio_file(wav_path, batch_size = batchsize, aggressiveness = aggressive, outlier_duration_ms = outlierdurms, outlier_batch_size = outlierbatchsize)
        print("dataset:", dataset)
def main():
    transcribe_file('why_should_one_halt.wav')


if __name__ == '__main__':
    main()
