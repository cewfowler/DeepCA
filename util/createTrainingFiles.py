import os;
import csv;
import json;
import argparse;
from pydub import AudioSegment;


# Write labeled data to a file
def writeLabeledData(data, file_to_write):
    f = open(file_to_write, 'w+');
    json.dump(data, f);
    f.close()


# Extract data from given TRN and store as new file
def extractData(file_to_read, file_to_write):
    data = [];
    ts = [];
    f = open(file_to_read, 'r');
    reader = csv.reader(f, delimiter='\t');

    # Read file and get data
    for row in reader:
        ts_len = len(row[0]);

        # Get the difference in the timestamps to 2 decimal places
        # String is read differently depending on length
        if (ts_len % 2 == 1):
            begin = float(row[0][:int(ts_len/2)]);
            end = float(row[0][int(ts_len/2+1):]);
            diff = round(end - begin, 2);

        else:
            begin = float(row[0][:int(ts_len/2)]);
            end = float(row[0][int(ts_len/2):]);
            diff = round(end - begin, 2);

        ts.append([begin, end]);
        data.append({"Duration": diff, "Text": row[2], "Begin": begin, "End": end});

    f.close();
    writeLabeledData(data, file_to_write);
    return ts;


# Splices a .wav file into smaller wav files using timestamps
# TODO: should spliced files be resampled to 16k?
def splice_wav(dir_to_read, dir_to_write, file_to_splice, ts):
    i = 1;
    base_name = os.path.splitext(file_to_splice)[0];
    audio = AudioSegment.from_wav(dir_to_read + '/' + base_name + '.wav');

    # Create directory for current audio file splicing
    if (not os.path.exists(dir_to_write + '/' + base_name)):
        os.makedirs(dir_to_write + '/' + base_name);
    folder = dir_to_write + '/' + base_name;

    # For each of the time stamps, get corresponding audio and save as new file
    for t in ts:
        t1 = t[0] * 1000;
        t2 = t[1] * 1000;
        print('Writing ' + folder + '/' + base_name + '_' + str(i) + '.wav');

        newAudio = audio[t1:t2];
        newAudio = newAudio.set_frame_rate(16000);

        newAudio.export(folder + '/' + base_name + '_' + str(i) + '.wav', format="wav");
        i = i + 1;


# Takes in TRN, data, audio, and spliced directories as arguments.
# Goes through a directory of TRN files and extracts data and timestamps.
# Then goes through a directory of corresponding .wav files and splices them
# into smaller files according to the timestamps.
def main():
    parser = argparse.ArgumentParser(description="""Splice wav file into smaller
        files based on time stamps found in a TRN file. Please note that audio files
        and their corresponding TRN files must have the same name""");
    parser.add_argument('--trn', help='Directory of the TRN files. Default: \'../SBC_TRN\'', default='../SBC_TRN');
    parser.add_argument('--data', help='Directory to put the extracted data files. Default: \'../labeled_data\'', default='../labeled_data');
    parser.add_argument('--audio', help='Directory of the audio files. Default: \'../SBC_Audio\'', default='../SBC_Audio');
    parser.add_argument('--spliced', help='Directory to put the spliced audio files. Default: \'../spliced_audio\'', default='../spliced_audio');
    args = parser.parse_args();

    for file in os.listdir(args.trn):
        writeFile = os.path.splitext(file)[0] + '_extracted.json';

        # If extracted data directory doesn't exist, create it
        if (not os.path.exists(args.data)):
            os.makedirs(args.data);

        ts = extractData(args.trn + '/' + file, args.data + '/' + writeFile);
        print('Successfully extraced data for ' + file);

        # If spliced audio directory doesn't exist, create it
        if (not os.path.exists(args.spliced)):
            os.makedirs(args.spliced);

        splice_wav(args.audio, args.spliced, file, ts);
        print('Successfully spliced wav file');

    return 0;

if __name__ == '__main__':
    main();
