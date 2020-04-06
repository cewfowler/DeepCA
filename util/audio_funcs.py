import os
import io
import tempfile
import collections
import wave
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio
from multiprocessing import cpu_count
# https://www.scivision.dev/python-windows-visual-c-14-required/ IF error trying to install webrtcvad
from webrtcvad import Vad

# This code was made referencing the open source project DeepSpeech by mozilla

# In order to get the wav file as dataset, must first split_audio_file.
# First, audio path is inserted into the transcribe_file function which eventually calls split_audio_file
# split_audio_file should take in same audio_path as transcribe_file. the audio_path is from AudioFile class which has many functions
# to make sure that a wav file is the correct rate, channels, width, format, etc. to be converted into the dataset


# Default settings
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)

# Without the training phase part.
def samples_to_mfccs(samples, sample_rate, train_phase= False, sample_id= None):
    # Window_size and stride gotten from config file and figuring out what the flags were
    spectrogram = contrib_audio.audio_spectrogram(samples, window_size= (16000 * (32/1000)), stride= (16000 * (20/1000)), magnitude_squared= True)
    # dct_coefficient_count gotten from config file (n_input = 26). same with upper frequency limit (16000 / 2)
    mfccs = contrib_audio.mfcc(spectrogram= spectrogram, sample_rate= sample_rate, dct_coefficient_count= 26, upper_frequency_limit= 8000)
    # reshape to [1, n_input]
    mfccs = tf.reshape(mfccs, [-1, 26])
    return mfccs, tf.shape(input=mfccs)[0]

# Used in pcm function and obtains the num sample size?
def get_num_samples(pcm_buffer_size, audio_format= DEFAULT_FORMAT):
    _, channels, width = audio_format
    return pcm_buffer_size // (channels * width)

# Used for obtaining the PCM (Pulse Code Modulation) duration typically read from a WAV file
def get_pcm_duration(pcm_buffer_size, audio_format= DEFAULT_FORMAT):
    return get_num_samples(pcm_buffer_size, audio_format) / audio_format[0]

# Used to convert the PCM to a numpy array
def pcm_to_np(audio_format, pcm_data):
    _, channels, width = audio_format
    if width not in [1, 2, 4]:
        raise ValueError('Unsupported width')
    dtype = [None, np.int8, np.int16, None, np.int32][width]
    samples = np.frombuffer(pcm_data, dtype= dtype)
    assert channels == 1
    samples = samples.astype(np.float32) / np.iinfo(dtype).max
    return np.expand_dims(samples, axis=1)

# Used for reading audio frames
def read_frames(wav_file, frame_duration_ms= 30, yield_remainder= False):
    if check_wav_audio_format(wav_file) == DEFAULT_FORMAT:
        audio_format = check_wav_audio_format(wav_file)
        frame_size = int(audio_format[0] * (frame_duration_ms / 1000.0))
        while True:
            try:
                data = wav_file.readframes(frame_size)
                if not yield_remainder and get_pcm_duration(len(data), audio_format) * 1000 < frame_duration_ms:
                    break
                yield data
            except EOFError:
                break

# Used for checking if wav audio is the default format
def check_wav_audio_format(wav_file):
    return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()

# Used for converting an audio file that isn't default format
def convert_audio(audio_path_source, audio_path_dest, file_type= None, audio_format= DEFAULT_FORMAT):
    sample_rate, channels, width = audio_format
    import sox
    transformer = sox.Transformer()
    transformer.set_output_format(file_type= file_type, rate = sample_rate, channels= channels, bits= width*8)
    transformer.build(audio_path_source, audio_path_dest)

# AudioFile class to be used for input
class AudioFile:
    def __init__(self, audio_path, as_path= False, audio_format= DEFAULT_FORMAT):
        self.audio_path = audio_path
        self.audio_format = audio_format
        self.as_path = as_path
        self.open_file = None
        self.tmp_file_open = None

    def __enter__(self):
        # if file ends with .wav
        if self.audio_path.endswith('.wav'):
            # 'rb' for reading and 'wb' for writing
            self.open_file = wave.open(self.audio_path, 'rb')
            # if the audio file meets the default format
            if check_wav_audio_format(self.open_file) == self.audio_format:
                if self.as_path:
                    self.open_file.close()
                    return self.audio_path
                return self.open_file
            # if the audio file is not the default format, start the process of converting it to default?
            self.open_file.close()
        _, self.tmp_file_path = tempfile.mkstemp(suffix='.wav')
        # Convert the audio into wav format with default format and return that
        convert_audio(self.audio_path, self.tmp_file_path, file_type= 'wav', audio_format= self.audio_format)
        if self.as_path:
            return self.tmp_file_path
        self.open_file = wave.open(self.tmp_file_path, 'rb')
        return self.open_file

    def __exit__(self, *args):
        if not self.as_path:
            self.open_file.close()
        if self.tmp_file_open is not None:
            os.remove(self.tmp_file_path)

# Vad_split for getting the segments needed in generating values? Possibly make aggressiveness None to pass in own
def vad_split(audio_frames, audio_format= DEFAULT_FORMAT, num_padding_frames= 10, threshold= 0.5, aggressiveness= 3):
    sample_rate, channels, width = audio_format
    if channels != 1:
        raise valueError('Requires mono channels')
    if width != 2:
        raise ValueError('Requires 16 bit samples')
    if sample_rate not in [8000, 16000, 32000, 64000]:
        raise ValueError('Unsupported sample rate')
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('Unsupported aggressiveness rate')
    ring_buffer = collections.deque(maxlen= num_padding_frames)
    triggered = False
    vad = Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('Unsupported frame duration')
        is_speech = vad.is_speech(frame, sample_rate)
        # If triggered is False
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        # If triggered is True
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), \
                    frame_duration_ms * max(0, frame_index - len(voiced_frames)), \
                    frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b''.join(voiced_frames), \
            frame_duration_ms * (frame_index - len(voiced_frames)), \
            frame_duration_ms * (frame_index + 1)

# Used in create_batch_set. Wraps a tensorflow dataset generator for catching its actual exceptions that would otherwise interrupt iteration
def remember_exception(iterable, exception_box= None):
    def iterate():
        try:
            yield from iterable()
        except StopIteration:
            return
        except Exception as ex:
            exception_box.exception = ex
    return iterable if exception_box is None else iterate

# Splitting up audio for dataset. Batchsize, aggressiveness, outliers left to none for our own custom values to be called in the input
def split_audio_file(audio_path, audio_format= DEFAULT_FORMAT, batch_size= None, aggressiveness= None, outlier_duration_ms= None, outlier_batch_size= None, exception_box= None):
    def generate_values():
        frames = read_frames(audio_path)
        segments = vad_split(frames, aggressiveness = aggressiveness)
        # Returns 1 as of here (successful)
        for segment in segments:
            segment_buffer, time_start, time_end = segment
            samples = pcm_to_np(audio_format, segment_buffer)
            yield time_start, time_end, samples

    def to_mfccs(time_start, time_end, samples):
        features, features_len = samples_to_mfccs(samples, audio_format[0])
        return time_start, time_end, features, features_len

    def create_batch_set(bs, criteria):
        return (tf.data.Dataset
                .from_generator(remember_exception(generate_values, exception_box),
                                output_types=(tf.int32, tf.int32, tf.float32))
                .map(to_mfccs, num_parallel_calls= tf.data.experimental.AUTOTUNE)
                .filter(criteria)
                .padded_batch(bs, padded_shapes=([], [], [None, 26], [])))
    nds = create_batch_set(batch_size, lambda start, end, f, fl: end - start <= int(outlier_duration_ms))
    ods = create_batch_set(outlier_batch_size, lambda start, end, f, fl: end - start > int(outlier_duration_ms))
    dataset = nds.concatenate(ods)
    # Overlaps the preprocessing and model execution of a training step // https://www.tensorflow.org/guide/data_performance
    dataset = dataset.prefetch(cpu_count())
    return dataset
