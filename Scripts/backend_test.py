# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:29:52 2018

@author: hazim

This is the local backend code test bench for API deployment.
Tested on:
    -Tensorflow 1.10
    -Keras 2.2.2
"""

## Import relevant libraries
import gc
import pickle
import numpy as np
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model
import warnings

## Instance variables
warnings.filterwarnings('ignore')
INPUT_SIZE = 700
text = 'cool'


def main():

    ## Loading the Keras Tokenizer sequence file
    with open('../Models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences([text])
    x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_test = x_test.astype(np.int64)

    ## Load the Keras-Tensorflow model
    model = load_model('../Models/mcp_erezeki_word_conv1D.hdf5')
    
    ## Predict using the loaded model
    print('Text: ',text )
    print('Predicted sentiment: ', 'Positive' if model.predict_classes(x_test)[0]==1 else 'Negative')
    print('Probability of positive sentiment: ',model.predict_proba(x_test)[0][1])
    print('Probability of negative sentiment:', model.predict_proba(x_test)[0][0])

if __name__ == '__main__':
    main()
    gc.collect()

