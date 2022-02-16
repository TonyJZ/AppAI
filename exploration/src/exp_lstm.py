#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import pickle

import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


import tensorflow.keras as keras
from appai_lib import my_io
from appai_lib import preprocessing
import os


sns.set(style="dark")

def evaluate(y,yhat):
    rmse = np.sqrt(np.mean(y - yhat)**2)
    return rmse


n_features = 1
n_steps = 10
epoc = 1
exp_name = "small_lstm_with_validatioin"
data_dir = "../data/results/" + exp_name +"/"
model_dir = "../data/models/" + exp_name + "/"


if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

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



in_file = "../data/processed/april_may_elapsed_filtered.csv"
all_data =my_io.load_data(in_file)
grouped = all_data.groupby("meter_id")
with PdfPages( "../reports/" + exp_name + ".pdf") as pdf:
    for name, group in grouped:
         group = group.sort_values(by=["create_date"])
         train_data, test_data = preprocessing.split_data(group,"2019-04-30 00:00:00")
         x_train, y_train = split_sequence(train_data.ap, n_steps)
         x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
         model = Sequential()

         model.add(LSTM(20, activation="relu", input_shape=(n_steps, n_features), kernel_initializer='random_uniform'))
         #model.add(Dense(20))
         #model.add(LSTM(50, activation="relu" ))


         #model.add(Dropout(0.3))
         model.add(Dense(1, kernel_initializer='random_uniform', activation='linear'))

         adam_optimizer = keras.optimizers.Adam(lr=0.001)
         model.compile(optimizer = adam_optimizer, loss='mse')
         
         es = EarlyStopping(monitor = 'val_loss',mode = "min", verbose = 1,patience = 20)
         mc = ModelCheckpoint(model_dir + str(name) + ".model", monitor="val_loss", mode = "min")
         cb_list = [es, mc]
         hst = model.fit(x_train, y_train, epochs=epoc,validation_split = 0.2,callbacks = cb_list , verbose=1)
         
         #hst = model.fit(x_train, y_train, epochs=epoc, verbose=1)
         hst_pd = pd.Series(hst.history['loss'])
         hst_pd.to_csv(data_dir+"history_"+str(name)+".csv")
         model.save(model_dir + str(name) + ".model", "wb")
         #predict
         x_test,y_test = split_sequence(test_data.ap, n_steps)
         x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
         y_train_hat = model.predict(x_train,verbose =1)

         train_result = pd.DataFrame()
         train_result["ds"] = train_data["create_date"][n_steps:].reset_index(drop = True)
         train_result["yhat"] = pd.Series(y_train_hat[:,0])
         train_result["y"] = pd.Series(y_train)
         train_result["ydiff"] = (train_result.yhat - train_result.y).abs()
         train_result.to_csv(data_dir +"train_" + str(name) +".csv")
         train_rmse = evaluate(train_result.yhat,train_result.y)

         yhat = model.predict(x_test,verbose=1)

         #prediction dataset
         result = pd.DataFrame()
         result["ds"] = test_data["create_date"][n_steps:].reset_index(drop = True)
         #result = result.reset_index()
         result["yhat"] = pd.Series(yhat[:,0])
         result["y"] = pd.Series(y_test)
         result["ydiff"] = (result.yhat - result.y).abs()
         result.to_csv(data_dir + str(name) + ".csv")

         #draw
         fig = plt.figure(figsize =(25,25))
         fig.suptitle("Meter id = " + str(name))

         #draw the training set
         ax = plt.subplot(4,1,1)
         ax.set_title("loss")
         sns.lineplot(y = hst.history['loss'], x = [ i for i in range(0,len(hst.history['loss']))])

         ax.set_ylabel("mse")
         ax.set_xlabel("epoc")

         ax = plt.subplot(4,1,2)
         ax.set_title("Training data, RMSE = " + str(train_rmse))

         sns.lineplot(ax = ax, y = "y", x = "ds", data = train_result)
         sns.lineplot(ax = ax, y = "yhat", x = "ds", data = train_result)
         sns.lineplot(ax = ax, y = "ydiff", x = "ds", data = train_result)

         ax.legend(labels = ["Real","Predicted","Error"])
         every_nth = train_data.create_date.count()//6
         for n, label in enumerate(ax.xaxis.get_ticklabels()):
             if n % every_nth != 0:
                 label.set_visible(False)


         ax.set_ylabel("ap")
         ax.set_xlabel("create_time")

         #draw the result
         ax = plt.subplot(4,1,3)
         rmse = evaluate(result.y, result.yhat)
         ax.set_title("RMSE = " + str(rmse))

         sns.lineplot(ax = ax, y="y", x ="ds" , data = result)
         sns.lineplot(ax = ax, y="yhat", x ="ds" , data = result)
         sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
         ax.legend(labels = ["Real","Predicted","Difference"])
         every_nth = 700
         for n, label in enumerate(ax.xaxis.get_ticklabels()):
             if n % every_nth != 0:
                label.set_visible(False)
         ax.set_ylabel("ap")
         ax.set_xlabel("create_time")

         #draw anomaly detection
         ax = plt.subplot(4,1,4)
         sns.lineplot(ax = ax, y="yhat", x ="ds" , data = result)
         std = result.yhat.std()
         ax.fill_between(result.ds, result.yhat - std * 4,result.yhat + std * 4, color="lightblue", alpha=0.2)

         outside = result[((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
         inside = result[~((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
         ax.set_title("Normal= " + str(inside.yhat.count()) + " Outlier = " + str(outside.yhat.count()) + " Outlier(%) =" + str(outside.yhat.count() * 100 /result.yhat.count()))
         sns.scatterplot(ax = ax, y = "y",x = "ds", data = inside)
         sns.scatterplot(ax = ax, y = "y",x = "ds", data = outside)
         ax.legend(labels=["Prediction","Threshold", "Normal","Outlier"])
         every_nth = 700
         for n, label in enumerate(ax.xaxis.get_ticklabels()):
             if n % every_nth != 0:
                label.set_visible(False)
         ax.set_ylabel("ap")
         ax.set_xlabel("create_time")
         pdf.savefig()
         plt.clf()
         plt.close()
         gc.collect()
         