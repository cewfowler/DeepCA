import csv;
import json;
import argparse;
from sklearn.preprocessing import LabelEncoder;


# Returns all unique characters found in given input file
def getCharactersFromTranscript(transcript):
    f = open(transcript, 'r');
    reader = csv.reader(f, delimiter='\t');
    chars_list = [];

    for row in reader:
        str = row[2];
        print(str);
        for c in str:
            try:
                chars_list.index(c);
            except:
                chars_list.append(c);

    f.close();

    return sorted(chars_list);


# Encode phonemes stored in p_file
# Returns y (transformed labels) and classes (list of classes/phonemes)
def encodePhonemes(p_file, verbose=False):
    f = open(p_file, 'r');
    p_list = f.read().splitlines()[1:-1];

    le = LabelEncoder();
    y = le.fit_transform(p_list);
    classes = list(le.classes_);

    if (verbose):
        print("Phonemes:")
        i = 1;
        for p in p_list:
            print(str(i) + ": " + p);
            i = i + 1;

        print(list(le.classes_));

    f.close();
    return y, classes;



# Takes in transcript file path
# Calls getCharactersFromTranscript and prints out all the unique, sorted chars
def main():
    parser = argparse.ArgumentParser(description="""Get unique characters from a
                                        transcript file""");
    parser.add_argument('--transcript', help="""Transcript file (including path)
            to search for unique characters""", default='./SBC_TRN/SBC001.trn');
    args = parser.parse_args();

    chars = getCharactersFromTranscript(args.transcript);
    print("Final sorted chars list:");
    for c in chars:
        print(c + '\n');


if __name__ == "__main__":
    main();
