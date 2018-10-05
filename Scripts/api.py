# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:52:53 2018

@author: hazim

POC API script for sentiment analysis.

Deployed using:
    -Tensorflow 1.10
    -Keras 2.2.2
    -Flask 1.0.2
    -PyMySQL 0.9.2

"""

import flask
import pickle
import pymysql
import numpy as np
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model
from flask import request, jsonify
import warnings

global tokenizer
global model
global text
global INPUT_SIZE

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['JSON_SORT_KEYS'] = False
warnings.filterwarnings('ignore')
tokenizer = None
model = None
text = None
INPUT_SIZE=700
    
@app.route('/api/v1/sentiment', methods=['GET'])
def api_sentiment():
    global text
    if 'text' in request.args:
        text = str(request.args['text'])
        results = predict(text)
        return(jsonify(results))
    else:
        return "Error: No text field provided. Please specify a text."

def predict(text):
    
    global tokenizer
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
    
    return({'Text':text, 'Sentiment': str(sentiment), 'Probability of positive sentiment': str(positive_probability), 'Probability of negative sentiment': str(negative_probabiltiy)})

@app.after_request
def save_to_db(response):
    global text
    if text != None:
        db = pymysql.connect("localhost","badrul","2018Mysql","sentiment" )
        cursor = db.cursor()
        sql = "INSERT INTO API_text(Text) VALUES ('%s')" % (text)
        cursor.execute(sql)
        db.commit()
        db.close()
        text=None
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
        
    app.run()

if __name__ == '__main__':
    main()