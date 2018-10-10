# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:33:40 2018

@author: hazim
"""

import flask
import pickle
import numpy as np
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model
from flask import request, jsonify
import warnings

global tokenizer
global model
global result
global INPUT_SIZE

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['JSON_SORT_KEYS'] = False
warnings.filterwarnings('ignore')
tokenizer = None
model = None
result = None
INPUT_SIZE=700
db_host = 'yourHost'
db_username = 'userName'
db_pass = 'userPass'
db_name = 'dbName'


## Main API get hook function
@app.route('/api/v1/sentiment', methods=['GET'])
def api_sentiment():
    global result
    if 'text' in request.args:
        text = str(request.args['text'])
        if text == '':
            return "Error: No text provideed. Please specify a text."
        result = predict(text)
        return(jsonify({'Text':result[0], 'Predicted sentiment': str(result[1]), 'Probability of positive sentiment': str(result[2]), 'Probability of negative sentiment': str(result[3])}))
    else:
        return "Error: No text field provided. Please specify a text."

def predict(text):
    global model
    
    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences([text])
    x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                     maxlen=INPUT_SIZE,
                                     padding='post')
    x_test = x_test.astype(np.int64)
    
    ## Predict using the loaded model
    sentiment = 'Positive' if model.predict_classes(x_test)[0]==1 else 'Negative'
    positive_probability = model.predict_proba(x_test)[0][1]
    negative_probabiltiy = model.predict_proba(x_test)[0][0]
    
    return([text, sentiment, positive_probability, negative_probabiltiy])

## Function to load after returning the response
@app.after_request
def save_to_db(response):

    if result[0] != '':
        print(request.environ.get('HTTP_X_REAL_IP', request.remote_addr))
        print("Testing")
    return response

def main():
    
    ## Load the Keras-Tensorflow model
    global model
    model = load_model('../Models/mcp_erezeki_word_conv1D.hdf5')
    model._make_predict_function()
    
    ## Loading the Keras Tokenizer sequence file
    global tokenizer
    with open('../Models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    app.run(host='localhost', port=5000)

if __name__ == '__main__':
    main()
