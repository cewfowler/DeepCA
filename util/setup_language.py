import csv;
import json;
import argparse;

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
                #print("Found " + c);
            except:
                chars_list.append(c);
                #print("Did not find " + c + ". Adding to list.")

    f.close();

    return sorted(chars_list);


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
