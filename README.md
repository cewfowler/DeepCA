# DeepCA #

## Creating Training Files ##
Conversation Analysis typically involves long, natural conversations which are inefficient to use for training a deep learning model. Therefore, we created a script to automatically splice large .wav files using corresponding .trn files, which contain the required timestamps. The script is found in the **_util_** folder and is called **_createTrainingFiles.py_**. Run `python3 util/createTrainingFiles.py -h` for more information on usage.

## Models ##
#### GRU Model ####
- The GRU Model consists of 3 non-recurrent layers and 2 GRU layers.


## Visualizing Models ##
- In order to visualize the models, the following dependencies must be installed first:
```
pip3 install keras
pip3 install ann_visualizer
pip install graphviz
pip install h5py
```
