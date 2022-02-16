#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""

import pandas as pd
import numpy as np
import datetime
import gc
import tensorflow as tf

import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import re
from datetime import timedelta
import dateutil.parser

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale
import scipy.signal as signal


import tensorflow.keras as keras
from appai_lib import preprocessing
import os

from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

exp_name = "cnaf_5m_ep_diff_july_rolling_section_2"
ex = Experiment(exp_name)
# ex.observers.append(MongoObserver.create(db_name="appai_sacred"))



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


def rolling_prediction(model,p_steps,n_steps,y_test):
    mean_errors = np.zeros(p_steps)
    i = 0
    yhats = list()
    while i < len(y_test) - (n_steps ):
        temp_ps = list()
        for j in range(0, p_steps):
            x = list(y_test.iloc[i:i + n_steps - j]) + temp_ps[0:j]   
            x = np.array(x)
            x = x.reshape((1, n_steps, 1))
            yhat = model.predict(x)
            temp_ps.append(yhat[0])
            yhats.append(yhat[0])    
            mean_errors[j] = mean_errors[j] + abs(yhat[0] - y_test.iloc[i + n_steps] )
            i = i + 1
            if i >= len(y_test) - (n_steps):
                break
            
    mean_errors = mean_errors / ((len(y_test - (n_steps + 1)))//n_steps) 
    return np.array(yhats), mean_errors    
            
                            


@ex.config
def cfg():
    # june_labeled_2
    # selected_meters = [1,5,19,32,50,52,66,73]#,19,20,21,22,26,32,33,40,42,43,50,52,53,58,60,61,65,66,67,68,69,72,73,76,77,80,106,210,245,249,343]
    # selected_meters = [19, 32, 52, 66, 73]
    selected_meters = [19]
    #june_labeled
    # selected_meters = [77, 65, 106, 210, 245, 249, 343]

    #selected_meters = [1, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
    #selected_meters = [293,85,409,245,543,249,114,405,375,79,77,
    #selected_meters = [159,438,311,238,491,13,365,265,248,591,343,106,210,184,592,164,102,371,274,348,443,121,81,257,411,532,562,53,65,427,108,404,206,396,80,161,549,509]
    #selected_meters = [73, 76, 77, 79, 80]
    #selected_meters = [4,11,12,77, 20,21,22]
    n_features = 1
    n_steps = 24
    p_steps = 15
    epoc = 200

    exp_name = exp_name
    data_dir = "../data/results/" + exp_name +"/"
    model_dir = "../data/models/" + exp_name + "/"
    prediction_column = "ep_diff"
    #in_dir = "../data/processed/june/"
    in_dir = "../data/processed/july_processed_ep_diff/"
    window_size = 3
    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + model_dir
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(in_dir):
        in_dir = os.path.abspath(os.path.dirname(__file__)) + "/" + in_dir

@ex.automain
def run(_log, selected_meters, n_features, n_steps, p_steps, epoc, exp_name, data_dir, model_dir, prediction_column, in_dir, window_size):
    #model
    in_file = [in_dir + str(i) + ".csv" for i in selected_meters]
    model = Sequential()
    model.add(LSTM(120, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform', return_sequences=True))
    model.add(LSTM(120, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))
    #model.add(Dense(28, activation="relu",  input_dim= 28, kernel_initializer='random_uniform'))
    #model.add(Dense(50))
    #model.add(Dense(50, activation="relu" ))
    #model.add(Dense(50, activation="relu" ))
    
    #model.add(Dropout(0.3))
    
    #model.add(LSTM(20, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))

    model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
    adam_optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer = adam_optimizer, loss='mse')

        
    for name in selected_meters:
        print (name)
        in_file = in_dir + str(name) + ".csv"
        meter_data = pd.read_csv(in_file)
        
        meter_data, _ = preprocessing.split_data(meter_data,"2019-07-20 00:00:00")
        train_data, test_data = preprocessing.split_data(meter_data,"2019-07-01 00:00:00" )
        #scale
        #train_x_max = train_data[prediction_column].max()
        #train_x_min = train_data[prediction_column].min()
        #train_x_delta = train_x_max - train_x_min
        #train_smooth = pd.Series(signal.medfilt(train_data[prediction_column], window_size))
        # prediction_column = "ep"
        prediction_column = "ep_diff"
        train_scale_factor = train_data[prediction_column].max() - train_data[prediction_column].min()
        train_min = train_data[prediction_column].min()
        train_data["scaled_prediction_column"] = pd.Series(minmax_scale(train_data[prediction_column]))
        
        # test_scale_factor =  test_data[prediction_column].max() - test_data[prediction_column].min()
        # test_min = test_data[prediction_column].min()
        test_scale_factor = train_scale_factor
        test_min = train_min
        # test_data["scaled_prediction_column"] = pd.Series(minmax_scale(test_data[prediction_column]))
        test_data["scaled_prediction_column"] = (test_data[prediction_column] - test_min)/test_scale_factor
        prediction_column = "scaled_prediction_column"
        
        #train_data["scaled_prediction_column"] = pd.Series(minmax_scale(train_data[prediction_column]))
        
        #x_train = train_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_train = train_data["scaled_prediction_column"].to_numpy()
        #print(x_train)
        #print(y_train)
        #x_train, y_train = add_time_feature(train_data, "scaled_prediction_column", n_steps)
        
        x_train, y_train = split_sequence(train_data[prediction_column] , n_steps)
        x_test,y_test = split_sequence(test_data[prediction_column], n_steps)
        
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
        
        es = EarlyStopping(monitor = 'val_loss',mode = "min", verbose = 1,patience = 20,min_delta = 0.0005)
        mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")
        
        cb_list = [es]
        hst = model.fit(x_train, y_train, epochs=epoc, validation_split = 0.2,callbacks = cb_list , verbose=1)
        hst_pd = pd.Series(hst.history['loss'])
        hst_pd.to_csv(data_dir+"history_"+str(name)+".csv")
        model.save(model_dir + str(name) + ".model")

        # model = keras.models.load_model(model_dir + str(name) + ".model")
        #predict
        #test_data["scaled_prediction_column"] = test_data[prediction_column].apply(lambda row:(row - train_x_min) / train_x_delta)
        #x_test = test_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_test = test_data["scaled_prediction_column"].to_numpy()
        
        #x_test,y_test = add_time_feature(test_data, "scaled_prediction_column", n_steps)
        
      
        
        y_train_hat = model.predict(x_train, verbose =1)
    
        train_result = pd.DataFrame()
        train_result["create_date"] = train_data.create_date[n_steps:].reset_index(drop = True)
        train_result["ds"] = train_data["elapsed_time"][n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:,0]) * train_scale_factor + train_min
        train_result["y"] = pd.Series(y_train) * train_scale_factor + train_min
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        train_result.to_csv(data_dir + "train_" + str(name) + ".csv")
        
        #threshold = train_result["y"].std() * 3
        #min_th = threshold / 3
        #max_th = threshold

        #y_labelhat = None
        #rolling_std = None
    
        yhat_single = model.predict(x_test)
        
        yhat, rolling_errors = rolling_prediction(model,p_steps,n_steps,test_data[prediction_column])
        
        ms_abs = abs(yhat[:,0] - y_test )

        yhat_s = pd.Series(ms_abs)
        section_errors = yhat_s.groupby(yhat_s.index // p_steps).mean()
        n_rows = 4
        n_min_section = section_errors.nsmallest(n_rows)
        sec_i = 0
        for i in n_min_section.index:
            sec_result = pd.DataFrame()
            sec_result["create_date"] = test_data.create_date[n_steps:].reset_index(drop=True)[i * p_steps: i * p_steps + p_steps]
            sec_result["ds"] =test_data.elapsed_time[n_steps:].reset_index(drop=True)[i * p_steps: i * p_steps + p_steps]
            sec_result["yhat_multistep"] =  pd.Series(yhat[:,0])[i * p_steps: i * p_steps + p_steps] * test_scale_factor + test_min
            sec_result["yhat_singlestep"] = pd.Series(yhat_single[:,0])[i * p_steps: i * p_steps + p_steps] * test_scale_factor + test_min
            sec_result["y"] = pd.Series(y_test)[i * p_steps: i * p_steps + p_steps] * test_scale_factor + test_min
            sec_result.to_csv(data_dir + "sec_"  + str(sec_i) + "_" + str(name) + ".csv")
            sec_i +=1
            #sec_result["ydiff"] = (sec_result.yhat_ - sec_result.y).abs()
            
        #prediction dataset
        result = pd.DataFrame()
        result["create_date"] = test_data.create_date[n_steps:].reset_index(drop=True)
        result["ds"] = test_data["elapsed_time"][n_steps:].reset_index(drop = True)
        #result = result.reset_index()
        result["yhat_singlestep"] = pd.Series(yhat_single[:,0]) * test_scale_factor + test_min
        result["yhat_multistep"] = pd.Series(yhat[:,0]) * test_scale_factor + test_min
        result["y"] = pd.Series(y_test) * test_scale_factor + test_min
        result["ydiff"] = (result.yhat_multistep - result.y).abs()
        
        rolling_result = pd.DataFrame()
        rolling_result["mean_error"] = pd.Series(rolling_errors)
        rolling_result["scale_factor"] = pd.Series(test_scale_factor, index=rolling_result.index) 
        rolling_result.to_csv(data_dir + "rolling_result_" + str(name) + ".csv")
        
        
        #if "label" in test_data:
        #    result["label"] = test_data["label"][n_steps:].reset_index(drop = True)
        
        
        #result["threshold_upper"] = result.apply(lambda row: row.yhat + threshold, axis = 1)
        #result["threshold_lower"] = result.apply(lambda row: row.yhat - threshold, axis = 1)
        
        #if y_labelhat is None :
        #    result["label_hat"] = result.apply(lambda row: False if (row.y < row.threshold_upper and row.y > row.threshold_lower) else True, axis = 1  )
        #else :
        #    result["label_hat"] = pd.Series(y_labelhat[:,0])

        #if rolling_std is not None:
        #    result["rolling_std"] = pd.Series(rolling_std[:,0])

        result.to_csv(data_dir + str(name) + ".csv")
        #K.clear_session()
        print("continuing experiment " + exp_name)

    from appai_lib import vis
    begin = datetime.datetime.now()
    import exp_cnaf_draw
    exp_cnaf_draw.draw(exp_name)
    end = datetime.datetime.now()
    print( end - begin )
    _log.info("Time taken  = " + str(end - begin) )
    ex.add_artifact("../reports/" + exp_name + ".pdf")



