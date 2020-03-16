import os;

def readTRN(file):
    ts = [];
    f = open(file, 'r');
    print('Chars: ' + f.read(8));
    print('Read line: ' + f.readline());
    return [];

def main():
    for file in os.listdir('./SBC_TRN'):
        timestamps = readTRN('./SBC_TRN/' + file);
        # Keep break in for testing, remove for running
        break;


    return 0;

if __name__ == '__main__':
    main();
