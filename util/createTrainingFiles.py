import os;
import csv;
import json;

def writeLabeledData(data, file_to_write):
    f = open(file_to_write, 'w+');
    json.dump(data, f);
    f.close()

def extractData(file_to_read, file_to_write):
    data = [];
    f = open(file_to_read, 'r'); #,newline='\r');
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

        data.append({"Duration": diff, "Text": row[2]});

    f.close();
    writeLabeledData(data, file_to_write)

def main():
    try:
        for file in os.listdir('./SBC_TRN'):
            writeFile = os.path.splitext(file)[0] + '_extracted.json';

            # If extracted data directory doesn't exist, create it
            if (not os.path.exists('./labeled_data')):
                os.makedirs('./labeled_data');

            extractData('./SBC_TRN/' + file, './labeled_data/' + writeFile);
            print('Successfully extraced data for ' + file);
    except:
        print("An error occurred!")


    return 0;

if __name__ == '__main__':
    main();
