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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector

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

exp_name = "lstm_sustained_12_24_7"
ex = Experiment(exp_name)
ex.observers.append(MongoObserver.create(db_name="appai_sacred"))

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

def batch_prediction(model, sequence, n_steps, n_features):
    x_test, y_test = split_sequence(sequence, n_steps)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    yhat = model.predict(x_test, verbose=1)
    y_test = y_test.reshape((y_test.shape[0], 1))
    return yhat, y_test

def batch_prediction_skip_anomaly_stationary_threshold(model, seq, n_steps, n_features, threshold):
    yhat = np.empty((len(seq)-n_steps, 1))
    yreal = np.empty((len(seq)-n_steps, 1))
    for i in range(len(seq)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(seq)-1:
            break
        # gather input and output parts of the pattern
        seq_x = seq.iloc[i:end_ix]
        x_test = np.array(seq_x)
        x_test = x_test.reshape((1, n_steps, n_features))
        
        y = model.predict(x_test,verbose=0)
        yhat[i] = y[0]
        yreal[i] = seq.iloc[end_ix]
        ydiff = abs(yhat[i] - yreal[i])
        ydiffs[i] = ydiff
        
        n_past_steps = 96
        if  i > n_past_steps:
            ydiffs_mean = np.mean(ydiffs[i-n_past_steps:i])
            ydiffs_std = np.std(ydiffs[i-n_past_steps:i])
        else:
            ydiffs_mean = np.mean(ydiffs[:i])
            ydiffs_std = np.std(ydiffs[:i])
        
        threshold = ydiffs_mean +  10 * ydiffs_std
        
        label_hat[i] = False
        if(ydiff > threshold):
            seq.iloc[end_ix] = yhat[i]
            label_hat[i] = True
        
    
    return yhat, yreal, label_hat

def batch_prediction_skip_anomaly_rolling_threshold(model, seq, n_steps, n_features, max_th, min_th):
    #fill the border for the rolling std
    yreal = np.zeros((len(seq)-n_steps, 1), dtype=np.float32)
    yhat = np.zeros((len(seq), 1), dtype=np.float32)
    yhat_label = np.zeros((len(seq)-n_steps, 1), dtype=bool)
    rolling_std = np.zeros((len(seq)-n_steps, 1), dtype=np.float32)
    
    for i in range(len(seq)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(seq)-1:
            break
        # gather input and output parts of the pattern
        seq_x = seq.iloc[i:end_ix]
        x_test = np.array(seq_x)
        x_test = x_test.reshape((1, n_steps, n_features))
        
        y = model.predict(x_test,verbose=0)
        yhat[end_ix] = y[0]
        yreal[i] = seq.iloc[end_ix]

        win_diff = yhat[i:end_ix] - x_test.reshape((n_steps, 1))
        rolling_thresh = np.std(win_diff) * 3
        thresh = rolling_thresh
        # rolling_std[i] = rolling_thresh

        if thresh > max_th :
            thresh = max_th
        if thresh < min_th :
            thresh = min_th

        rolling_std[i] = thresh

        ydiff = yhat[end_ix] - yreal[i]
        signRst = np.sign(ydiff)
        ydiff = abs(ydiff)
        if(ydiff[0] > thresh):
            yhat[end_ix] = yhat[end_ix] - signRst[0]*thresh*0.8    #softly dragging to real data
            seq.iloc[end_ix] = yhat[end_ix]
            yhat_label[i] = True
    
    yhat = yhat[n_steps:len(seq)]
    
    return yhat, yreal, yhat_label, rolling_std

def postprocessing_label_confirm(yhat, yreal, yhat_label, n_steps, max_th, min_th):
    length = len(yhat)
    ydiff = yhat - yreal

    for i in range(length):
        # find the end of this pattern
        j = i  
        end_ix = j + n_steps
        # check if we are beyond the sequence
        if end_ix > length-1:
            break
        # gather input and output parts of the pattern
        seq_x = ydiff[j:end_ix]

        win_diff = seq_x.reshape((n_steps, 1))
        rolling_thresh = np.std(win_diff) * 3
        thresh = rolling_thresh
        # rolling_std[i] = rolling_thresh

        if thresh > max_th :
            thresh = max_th
        if thresh < min_th :
            thresh = min_th

        dy = abs(ydiff[i])
        if yhat_label[i] == True:
            if dy < thresh:
                yhat_label[i] = False
        else :  #has some problems in the FN correction
            if dy > thresh:
                yhat_label[i] = True
    
    return yhat_label


@ex.config
def cfg():
    # june_labeled_2
    #selected_meters = [1,3,4,5,12,14]#,19,20,21,22,26,32,33,40,42,43,50,52,53,58,60,61,65,66,67,68,69,72,73,76,77,80,106,210,245,249,343]
    
    #june_labeled
    # selected_meters = [77, 65, 106, 210, 245, 249, 343]

    #selected_meters = [1, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
    #selected_meters = [293,85,409,245,543,249,114,405,375,79,77,
    #selected_meters = [159,438,311,238,491,13,365,265,248,591,343,106,210,184,592,164,102,371,274,348,443,121,81,257,411,532,562,53,65,427,108,404,206,396,80,161,549,509]
    #selected_meters = [73, 76, 77, 79, 80]
    #selected_meters = [4,11,12,77, 20,21,22]
    selected_meters = [1, 3, 19, 21, 68, 50]
    n_features = 1
    n_steps = 12 * 24 * 7
    epoc = 200

    exp_name = exp_name
    data_dir = "../data/results/" + exp_name +"/"
    model_dir = "../data/models/" + exp_name + "/"
    prediction_column = "ep_diff"
    #in_dir = "../data/processed/june/"
    #in_dir = "../data/processed/june_labeled_2/"
    in_dir = "../data/processed/july_processed/"
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
def run(_log, selected_meters, n_features, n_steps, epoc, exp_name, data_dir, model_dir, prediction_column, in_dir, window_size):
    #model
    in_file = [in_dir + str(i) + ".csv" for i in selected_meters]
    model = Sequential()
    model.add(LSTM(200,activation = "relu", input_shape(n_steps,n_features)))
    model.add(RepeatVector(7))
    model.add(LSTM(200, activation = "relu", return_sequences = True))
    model.add(TimeDistributed(Dense(100,activation = "relu")))
    model.add(TimeDistributed(Dense(1)))
    
    #model.add(LSTM(80, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))
    
    #model.add(Dense(28, activation="relu",  input_dim= 28, kernel_initializer='random_uniform'))
    #model.add(Dense(50))
    #model.add(Dense(50, activation="relu" ))
    #model.add(Dense(50, activation="relu" ))
    
    #model.add(Dropout(0.3))
    
    #model.add(LSTM(20, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))

    #model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))
    adam_optimizer = keras.optimizers.Adam(lr=0.005)
    model.compile(optimizer = adam_optimizer, loss='mse')

        
    for name in selected_meters:
        print (name)
        in_file = in_dir + str(name) + ".csv"
        meter_data = pd.read_csv(in_file)
        assert not np.any(np.isnan(meter_data[prediction_column])) 
        
        train_data, test_data = preprocessing.split_data(meter_data,"2019-04-30 24:00:00")
        test_data, _ = preprocessing.split_data(test_data,"2019-06-09 11:15:00")
        print("train range " + str(train_data.create_date.iloc[0]) + "  ---  " + str(train_data.create_date.iloc[-1]))
        print("test range " + str(test_data.create_date.iloc[0]) + "  ---  " + str(test_data.create_date.iloc[-1]))
        
        #scale
        train_x_max = train_data[prediction_column].max()
        train_x_min = train_data[prediction_column].min()
        train_x_delta = train_x_max - train_x_min
        train_smooth = pd.Series(signal.medfilt(train_data[prediction_column], window_size))
        train_data["scaled_prediction_column"] = pd.Series(minmax_scale(train_smooth))
        #train_data["scaled_prediction_column"] = pd.Series(minmax_scale(train_data[prediction_column]))
        
        x_train, y_train = split_sequence(train_data["scaled_prediction_column"] , n_steps)
        #x_train = train_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_train = train_data["scaled_prediction_column"].to_numpy()
        #print(x_train)
        #print(y_train)
        #x_train, y_train = add_time_feature(train_data, "scaled_prediction_column", n_steps)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
        
        es = EarlyStopping(monitor = 'val_loss',mode = "min", verbose = 1,patience = 20,min_delta = 0.00005)
        mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")
        
        cb_list = [es]
        
        hst = model.fit(x_train, y_train, epochs=epoc,callbacks = cb_list , validation_split = 0.2, verbose=1)
        hst_pd = pd.Series(hst.history['loss'])
        hst_pd.to_csv(data_dir+"history_"+str(name)+".csv")
        model.save(model_dir + str(name) + ".model")

        # model = keras.models.load_model(model_dir + str(name) + ".model")
        #predict
        test_data["scaled_prediction_column"] = test_data[prediction_column].apply(lambda row:(row - train_x_min) / train_x_delta)
        #x_test = test_data[["workday","weekday","hour","minute"]].to_numpy()
        #y_test = test_data["scaled_prediction_column"].to_numpy()
        x_test, y_test = split_sequence(test_data["scaled_prediction_column"], n_steps)
        #x_test,y_test = add_time_feature(test_data, "scaled_prediction_column", n_steps)
        
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
        y_train_hat = model.predict(x_train, verbose =1)
    
        train_result = pd.DataFrame()
        train_result["create_date"] = train_data.create_date[n_steps:].reset_index(drop = True)
        train_result["ds"] = train_data["elapsed_time"][n_steps:].reset_index(drop = True)
        train_result["yhat"] = pd.Series(y_train_hat[:,0])
        train_result["y"] = pd.Series(y_train)
        train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
        train_result.to_csv(data_dir + "train_" + str(name) +".csv")
        threshold = train_result["y"].std() * 1
        
        min_th = threshold / 3
        max_th = threshold

        y_labelhat = None
        rolling_std = None
    
        # yhat, yreal = batch_prediction(model, test_data["scaled_prediction_column"], n_steps, n_features)
        # yhat, yreal = batch_prediction_skip_anomaly_stationary_threshold(model, test_data["scaled_prediction_column"], n_steps, n_features, threshold)
        #yhat, yreal, y_labelhat, rolling_std = batch_prediction_skip_anomaly_rolling_threshold(model, test_data["scaled_prediction_column"], n_steps, n_features, max_th, min_th)

        # post-confirm the anmolies
        #y_labelhat = postprocessing_label_confirm(yhat, yreal, y_labelhat, n_steps, max_th, min_th)
        
        yhat = model.predict(x_test)
        
        
        
        #prediction dataset
        result = pd.DataFrame()
        result["create_date"] = train_data.create_date[n_steps:].reset_index(drop = True)
        result["ds"] = test_data["elapsed_time"][n_steps:].reset_index(drop = True)
        #result = result.reset_index()
        result["yhat"] = pd.Series(yhat[:,0])
        result["y"] = pd.Series(y_test)
        result["ydiff"] = (result.yhat - result.y).abs()
        
        #simple thresholding
        result["threshold_upper"] = result.apply(lambda row: row.yhat + threshold, axis = 1)
        result["threshold_lower"] = result.apply(lambda row: row.yhat - threshold, axis = 1)

        result["label_hat"] = result.apply(lambda row: False if (row.y < row.threshold_upper and row.y > row.threshold_lower) else True, axis = 1  )
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

        # break
    

    from appai_lib import vis
    begin = datetime.datetime.now()
    vis.vis_result(exp_name)
    end = datetime.datetime.now()
    print( end - begin )
    _log.info("Time taken  = " + str(end - begin) )
    ex.add_artifact("../reports/" + exp_name + ".pdf")



