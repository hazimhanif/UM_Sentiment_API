# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:29:52 2018

@author: hazim

This is the local backend code test bench for API deployment.

"""

## Import relevant libraries
import gc
import pickle
import numpy as np
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model

## Instance variables
INPUT_SIZE = 700
text = 'I love this product very much. This is the best product ever!'


def main():

    ## Loading the Keras Tokenizer sequence file
    with open('../Models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences(text)
    x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_test = x_test.astype(np.int64)
    

    ## Load the Keras-Tensorflow model
    model = load_model('../Models/mcp_erezeki_word_conv1D.hdf5')
    
    ## Evaluate the model
    print(model.predict_classes(x_test))


if __name__ == "main":
    main()
    gc.collect()