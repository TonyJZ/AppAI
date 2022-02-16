#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""


# Set up the experiment
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

exp_name = "lstm_easy"
ex = Experiment(exp_name)
ex.observers.append(MongoObserver.create(db_name="appai_sacred"))


import pandas as pd
import numpy as np

import datetime
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

from sklearn.preprocessing import minmax_scale

import tensorflow.keras as keras
from appai_lib import preprocessing
import os

def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse



def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)



@ex.config
def config():
    n_features = 2
    n_steps = 28
    epoc = 30
    data_dir = "../data/results/" + exp_name +"/"
    model_dir = "../data/models/" + exp_name + "/"
    prediction_column = "ep_diff"
    in_dir = "../data/processed/june/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    selected_meters = [1]#, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
#selected_meters = [73, 76, 77, 79, 80]
#in_files = [in_dir + f for f in os.listdir(in_dir) if int(f.split(".")[0]) in selected_meters]

@ex.automain
def run(_log,n_features, n_steps, epoc,data_dir,model_dir,prediction_column,in_dir,selected_meters):
    in_file = [in_dir + str(i) + ".csv" for i in selected_meters]
    begin = datetime.datetime.now()
    
    #model
    model = Sequential()
    model.add(LSTM(20, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))
    #model.add(LSTM(50))
    #model.add(Dense(20))
    #model.add(Dense(50, activation="relu" ))
    #model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
    adam_optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer = adam_optimizer, loss='mse')
    
    for name in selected_meters:
        print (name)
        in_file = in_dir + str(name) + ".csv"
        meter_data = pd.read_csv(in_file)
        train_data, test_data = preprocessing.split_data(meter_data,"2019-04-30 00:00:00")
        train_x_max = train_data[prediction_column].max()
        train_x_min = train_data[prediction_column].min()
        train_x_delta = train_x_max - train_x_min
        scaled_train = pd.Series(minmax_scale(train_data[prediction_column]))
        x_train_ap, y_train = split_sequence(scaled_train, n_steps)
        x_train_workday,y_train_workday = split_sequence(train_data.workday, n_steps)
        x_train = np.dstack((x_train_ap,x_train_workday))
        
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    
        hst = model.fit(x_train, y_train, epochs=epoc, verbose=1)
        hst_pd = pd.Series(hst.history['loss'])
        hst_pd.to_csv(data_dir+"history_"+str(name)+".csv")
        model.save(model_dir + str(name) + ".model", "wb")
        #predict
        scaled_test = test_data[prediction_column].apply(lambda row: (row - train_x_min)/ train_x_delta)
        x_test_ap,y_test = split_sequence(scaled_test, n_steps)
        x_test_workday,y_test_workday = split_sequence(test_data.workday, n_steps)
        x_test = np.dstack((x_test_ap, x_test_workday))
        #x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],n_features))
        y_train_hat = model.predict(x_train,verbose =1)
    
        train_result = pd.DataFrame()
        train_result["ds"] = train_data["create_date"][n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:,0])
        train_result["y"] = pd.Series(y_train)
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        train_result.to_csv(data_dir +"train_" + str(name) +".csv")
    
        yhat = model.predict(x_test,verbose=1)
    
        #prediction dataset
        result = pd.DataFrame()
        result["ds"] = test_data["create_date"][n_steps:].reset_index(drop = True)
        #result = result.reset_index()
        result["yhat"] = pd.Series(yhat[:,0])
        result["y"] = pd.Series(y_test)
        result["ydiff"] = (result.yhat - result.y).abs()
        result.to_csv(data_dir + str(name) + ".csv")
        #K.clear_session()
        #gc.collect()
        print("continuing experiment " + exp_name)
    end = datetime.datetime.now()
    print( end - begin )
    _log.info("Time taken = " + str(end - begin))
    from appai_lib import vis
    vis.vis_result(exp_name)
    ex.add_artifact( "../reports/" + exp_name + ".pdf")