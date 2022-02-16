#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""

import ray
#ray.init(redis_address="10.10.10.63:6379",ignore_reinit_error = True)
ray.init(ignore_reinit_error=True)
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale

import tensorflow.keras as keras
from appai_lib import preprocessing
import os

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
exp_name = "sigmoud_lstm_without_ray"
ex = Experiment(exp_name)
# ex.observers.append(MongoObserver.create(db_name="appai_sacred"))


    
def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse

def add_time_feature(data_frame, prediction_column, n_steps):
    sequence = data_frame[prediction_column]

    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix]
        seq_x = seq_x.append(pd.Series([data_frame.workday.iloc[end_ix], data_frame.weekday.iloc[end_ix], data_frame.hour.iloc[end_ix],data_frame.minute.iloc[end_ix]]), ignore_index = True)
        #seq_x = seq_x.reset_index()
        X.append(seq_x)
        #X = X.reset_index()
        y.append(seq_y)
    return np.array(X), np.array(y)


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
def cfg():
    selected_meters = [1, 3, 4, 5]#, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
    #selected_meters = [293,85,409,245,543,249,114,405,375,79,77,
    #selected_meters = [159,438,311,238,491,13,365,265,248,591,343,106,210,184,592,164,102,371,274,348,443,121,81,257,411,532,562,53,65,427,108,404,206,396,80,161,549,509]
    #selected_meters = [73, 76, 77, 79, 80]
    #selected_meters = [4,11,12,77, 20,21,22]
    n_features = 1
    n_steps = 28
    epoc = 200
    num_layers = 3
    exp_name = exp_name
    data_dir = "../data/results/" + exp_name +"/"
    model_dir = "../data/models/" + exp_name + "/"
    prediction_column = "ep_diff"
    in_dir = "../data/processed/june/"

    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(in_dir):
        in_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + in_dir


@ray.remote
class Model():
    def __init__(self,n_features, n_steps, epoc,prediction_column):
        model = Sequential()
        model.add(Dense(28, activation="relu",  input_dim= 28, kernel_initializer='random_uniform'))
        model.add(Dense(50, activation="relu",  kernel_initializer='random_uniform'))
        model.add(Dense(50, activation="relu",  kernel_initializer='random_uniform'))
        model.add(Dense(50, activation="relu",  kernel_initializer='random_uniform'))
        
        model.add(Dense(1, kernel_initializer='random_uniform', activation='relu'))
        #model.add(Dense(1, kernel_initializer='random_uniform', activation='relu'))
        adam_optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer = adam_optimizer, loss='mse')
        self.model = model
        self.n_features = n_features
        self.n_steps = n_steps
        self.epoc = epoc
        self.prediction_column = prediction_column
    
    @ray.method(num_return_vals=3)
    def train_and_predict(self,train_data,test_data, name):
        #scale
        print("training  meter " + str(name))
        train_x_max = train_data[self.prediction_column].max()
        train_x_min = train_data[self.prediction_column].min()
        train_x_delta = train_x_max - train_x_min
        train_data["scaled_prediction_column"] = pd.Series(minmax_scale(train_data[self.prediction_column]))
        #train_data["scaled_prediction_column"]
        x_train, y_train = split_sequence(train_data["scaled_prediction_column"] , self.n_steps)
        #x_train = train_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_train = train_data["scaled_prediction_column"].to_numpy()
        #print(x_train)
        #print(y_train)
        #x_train, y_train = add_time_feature(train_data, "scaled_prediction_column", n_steps)
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
        
        es = EarlyStopping(monitor = 'val_loss', mode = "min", verbose = 0, patience = 20, restore_best_weights = True)
        #mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")
        
        #cb_list = [es, mc]
        cb_list = [es]
        print("starting training " + str(name))
        hst = self.model.fit(x_train, y_train, epochs=self.epoc,validation_split = 0.20 ,callbacks = cb_list , verbose=0)
        print("training complete " + str(name))
        hst_pd = pd.Series(hst.history['loss'])
        
        #hst_pd.to_csv(data_dir+"history_"+str(name)+".csv")
        
        #model.save(model_dir + str(name) + ".model", "wb")
        #predict
        test_data["scaled_prediction_column"] = test_data[self.prediction_column].apply(lambda row: (row - train_x_min)/ train_x_delta)
        #x_test = test_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_test = test_data["scaled_prediction_column"].to_numpy()
        x_test,y_test = split_sequence(test_data["scaled_prediction_column"], self.n_steps)
        #x_test,y_test = add_time_feature(test_data, "scaled_prediction_column", n_steps)
        
        #x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],n_features))
        y_train_hat = self.model.predict(x_train, verbose = 0)
    
        train_result = pd.DataFrame()
        train_result["ds"] = train_data["elapsed_time"][self.n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:,0])
        train_result["y"] = pd.Series(y_train)
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        #train_result.to_csv(data_dir + "train_" + str(name) +".csv")
    
        yhat = self.model.predict(x_test,verbose = 0)
    
        #prediction dataset
        result = pd.DataFrame()
        result["ds"] = test_data["elapsed_time"][self.n_steps:].reset_index(drop = True)
        #result = result.reset_index()
        result["yhat"] = pd.Series(yhat[:,0])
        result["y"] = pd.Series(y_test)
        result["ydiff"] = (result.yhat - result.y).abs()
        #result.to_csv(data_dir + str(name) + ".csv")
        print("returning result" + str(name))
        return result, hst_pd, train_result
        
        
        
@ex.automain
def run(_log, selected_meters, n_features, n_steps, epoc, num_layers, exp_name, data_dir, model_dir, prediction_column, in_dir):
    #model
    begin = datetime.datetime.now()
    in_file = [in_dir + str(i) + ".csv" for i in selected_meters]
    results = []
    for name in selected_meters:
        print (name)
        in_file = in_dir + str(name) + ".csv"
        meter_data = pd.read_csv(in_file)
        train_data, test_data = preprocessing.split_data(meter_data,"2019-04-30 00:00:00")
        model = Model.remote(n_features, n_steps, epoc,prediction_column)
        result,hst,train_result = model.train_and_predict.remote(train_data,test_data, name)
        results.append({"result":result, "hst": hst, "train_result" : train_result,"name":name})
        #results.append(results)
    for r in results:
        result = ray.get(r["result"])
        hst = ray.get(r["hst"])
        train_result = ray.get(r["train_result"])
        
        #hst_pd = pd.Series(hst.history['loss'])
        hst.to_csv(data_dir+"history_"+str(r["name"])+".csv")
        train_result.to_csv(data_dir + "train_" + str(r["name"]) +".csv")
        result.to_csv(data_dir + str(r["name"]) + ".csv")
        
    end = datetime.datetime.now()
    print("total time " + str( end - begin ))
    
    #from appai_lib import vis
    #begin = datetime.datetime.now()
    #vis.vis_result(exp_name)
    #end = datetime.datetime.now()
    #print( end - begin )
    #_log.info("Time taken  = " + str(end - begin) )
    #ex.add_artifact("../reports/" + exp_name + ".pdf")



