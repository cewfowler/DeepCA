# DeepCA #

## Creating Training Files ##
Conversation Analysis typically involves long, natural conversations which are inefficient to use for training a deep learning model. Therefore, we created a script to automatically splice large .wav files using corresponding .trn files, which contain the required timestamps. The script is found in the **_util_** folder and is called **_createTrainingFiles.py_**. Run `python3 util/createTrainingFiles.py -h` for more information on usage.

## Models ##
#### GRU Model ####
- The GRU Model consists of 3 non-recurrent layers and 2 GRU layers.
