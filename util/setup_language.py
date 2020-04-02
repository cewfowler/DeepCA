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


def encodePhonemes(p_file):
    #le = LabelEncoder();
    f = open(p_file, 'r');
    p_list = f.readlines();
    p_list = p_list[1:-1];
    
    print("Phonemes:")
    for p in p_list:
        print("This one: " + p)

    f.close();



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
