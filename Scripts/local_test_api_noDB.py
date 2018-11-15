# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:33:40 2018

@author: hazim
"""

import flask
import pickle
import numpy as np
import pymysql
import os
from tensorflow.python.keras.preprocessing import sequence as keras_seq
from tensorflow.python.keras.models import load_model
from flask import request, jsonify
import warnings

global tokenizer
global pred_models 
global result
global INPUT_SIZE
global error

app = flask.Flask(__name__)
app.config["DEBUG"] = False
app.config['JSON_SORT_KEYS'] = False
warnings.filterwarnings('ignore')
tokenizer = None
error = None

pred_models = {}
INPUT_SIZE = {'word2seq_cnn':700,
               'word2seq_cnn_birnn_bilstm':100,
               'word2seq_cnn_lstm':500,
               'word2seq_lstm':100,
               'word2vec_cnn':700,
               'word2vec_cnn_birnn_bilstm':100,
               'word2vec_cnn_lstm':500,
               'word2vec_lstm':100}

table_name = {'word2seq_cnn':'Word2Seq_CNN',
              'word2seq_cnn_birnn_bilstm':'Word2Seq_CNN_BiRNN_BiLSTM',
              'word2seq_cnn_lstm':'Word2Seq_CNN_LSTM',
              'word2seq_lstm':'Word2Seq_LSTM',
              'word2vec_cnn':'Word2Vec_CNN',
              'word2vec_cnn_birnn_bilstm':'Word2Vec_CNN_BiRNN_BiLSTM',
              'word2vec_cnn_lstm':'Word2Vec_CNN_LSTM',
              'word2vec_lstm':'Word2Vec_LSTM'}

WORDS_SIZE = 8000
db_host = 'yourHost'
db_username = 'userName'
db_pass = 'userPass'
db_name = 'dbName'
retName = ['Predicted sentiment','Probability of positive sentiment','Probability of negative sentiment']

## Main API get hook function
@app.route('/api/v1/sentiment', methods=['GET'])
def api_sentiment():
    global error
    error = False
    
    if 'text' in request.args:
        text = str(request.args['text'])
        if text == '':
            return "Error: No text provideed. Please specify a text."
        result = predict(text)
        return(jsonify(result))
    else:
        error = True
        return "Error: No text field provided. Please specify a text."

def predict(text):
    global pred_models
    return_dict={}
    
    ## Tokkenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences([text])
    return_dict.update({'Text':text})
    
    for model in [model[:-5]for model in os.listdir('../Models')[1:]]:
        x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                         maxlen=INPUT_SIZE[model],
                                         padding='post')
        x_test = x_test.astype(np.int64)

        ## Predict using the loaded model
        sentiment = 'Positive' if pred_models[model].predict_classes(x_test)[0]==1 else 'Negative'
        positive_probability = pred_models[model].predict_proba(x_test)[0][1]
        negative_probability = pred_models[model].predict_proba(x_test)[0][0]
        #save_to_db(model, text, sentiment, positive_probability, negative_probability)
        
        return_dict.update({table_name[model].replace('_',' '): 
            {retName[0]:str(sentiment), 
             retName[1]:str(positive_probability), 
             retName[2]:str(negative_probability)}})
    
    return(return_dict)

def save_to_db(model, text, sentiment, positive_probability, negative_probability):
    if error != True:
        db = pymysql.connect(db_host,db_username,db_pass,db_name)
        cursor = db.cursor()
        sql = "INSERT INTO %s(Text,Text_hash,Pred_sentiment,Prob_positive,Prob_negative,IP_address) VALUES ('%s', sha1(Text), '%s', %f, %f, '%s')" % (table_name[model], text, sentiment, positive_probability, negative_probability, request.environ.get('HTTP_X_REAL_IP', request.remote_addr))
        cursor.execute(sql)
        db.commit()
        db.close()

def main():
    ## Load the Keras-Tensorflow models into a dictionary
    global pred_models 
    
    pred_models={'word2seq_cnn' : load_model('../Models/word2seq_cnn.hdf5'),
    'word2seq_lstm' : load_model('../Models/word2seq_lstm.hdf5'),
    'word2seq_cnn_lstm' : load_model('../Models/word2seq_cnn_lstm.hdf5'),
    'word2seq_cnn_birnn_bilstm' : load_model('../Models/word2seq_cnn_birnn_bilstm.hdf5'),
    'word2vec_cnn' : load_model('../Models/word2vec_cnn.hdf5'),
    'word2vec_lstm' : load_model('../Models/word2vec_lstm.hdf5'),
    'word2vec_cnn_lstm' : load_model('../Models/word2vec_cnn_lstm.hdf5'),
    'word2vec_cnn_birnn_bilstm': load_model('../Models/word2vec_cnn_birnn_bilstm.hdf5')}
    
    ## Make prediction function
    for model in [model[:-5]for model in os.listdir('../Models')[1:]]:
        pred_models[model]._make_predict_function()
    
    ## Loading the Keras Tokenizer sequence file
    global tokenizer
    with open('../Models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    app.run(host='localhost', port=5000)

if __name__ == '__main__':
    main()
