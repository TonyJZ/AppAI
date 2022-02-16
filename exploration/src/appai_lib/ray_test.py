#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:39:52 2019

@author: Toukir Imam (toukir@appropolis.com)
"""

import ray
#ray.init(redis_address="10.10.10.63:6379",ignore_reinit_error = True)
ray.init(ignore_reinit_error=True)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import matplotlib
from fpdf import FPDF
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import datetime
sns.set_style("dark")
import ray

def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse

@ray.remote
def plot(result,train_result,hst, name,sl):
    fig = plt.figure(figsize =(25,26))
    #fig = plt.figure()
    fig.suptitle("Meter id = " + str(name))
    #draw the training set
    begin = datetime.datetime.now()
    ax = plt.subplot(5,1,1)
    ax.set_title("loss")
    sns.lineplot(y = np.sqrt(hst[1]), x = [ i for i in range(0,len(hst[1]))])
    ax.set_ylabel("rmse")
    ax.set_xlabel("epoc")
    train_rmse = evaluate(train_result.yhat,train_result.y)
    print ("epoc time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    ax = plt.subplot(5,1,2)
    ax.set_title("Training data, RMSE = " + str(train_rmse))
    
    #plt.plot(train_result.ds,train_result.y)
    #plt.plot(train_result.ds,train_result.yhat)
    #plt.plot(train_result.ds,train_result.ydiff)
    sns.lineplot(ax = ax, y = "y", x = "ds", data = train_result)
    sns.lineplot(ax = ax, y = "yhat", x = "ds", data = train_result)
    sns.lineplot(ax = ax, y = "ydiff", x = "ds", data = train_result)


    ax.legend(labels = ["Real","Predicted"])
    #every_nth = train_result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)


    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")

    #draw the result
    print ("training result time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    
    ax = plt.subplot(5,1,3)
    rmse = evaluate(result.y, result.yhat)
    ax.set_title("RMSE = " + str(rmse))

    sns.lineplot(ax = ax, y="y", x ="ds" , data = result)
    sns.lineplot(ax = ax, y="yhat", x ="ds" , data = result)
    #sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
    
    #plt.plot(result.ds,result.y)
    #plt.plot(result.ds,result.yhat)
    #plt.plot(result.ds,result.ydiff)
    ax.legend(labels = ["Real","Predicted","Difference"])
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)
    #ax.set_ylabel("ap")
    #ax.set_xlabel("create_time")
    
    #draw the Error
    print ("prediction time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    ax = plt.subplot(5,1,4)
    rmse = evaluate(result.y, result.yhat)
    ax.set_title("Difference, RMSE = " + str(rmse))
    sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = result)
    
    #plt.plot(result.ds,result.y)
    #plt.plot(result.ds,result.yhat)
    #plt.plot(result.ds,result.ydiff)
    ax.legend(labels = ["Difference"])
    #every_nth = result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #        label.set_visible(False)
    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")
    
    #draw anomaly detection
    print ("prediction diff time " + str(datetime.datetime.now() - begin))
    begin = datetime.datetime.now()
    
    
    ax = plt.subplot(5,1,5)
    #plt.plot(result.ds, result.yhat)
    sns.lineplot(ax = ax, y="yhat",x = "ds", data = result)
    std = result.yhat.std()
    ax.fill_between(result.ds, result.yhat - std * 4,result.yhat + std * 4, color="lightblue", alpha=0.2)

    outside = result[((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
    inside = result[~((result["y"] > result.yhat + std * 4) | (result["y"] < result.yhat - std * 4))]
    ax.set_title("Normal= " + str(inside.yhat.count()) + " Outlier = " + str(outside.yhat.count()) + " Outlier(%) =" +   str(outside.yhat.count() * 100 /result.yhat.count()))
    sns.scatterplot(ax = ax, y = "y",x = "ds", data = inside, color="green")
    sns.scatterplot(ax = ax, y = "y",x = "ds", data = outside, color= "red")
    ax.legend(labels=["Prediction","Threshold", "Normal","Outlier"])
    #every_nth = result.ds.count()//6
    #for n, label in enumerate(ax.xaxis.get_ticklabels()):
    #    if n % every_nth != 0:
    #       label.set_visible(False)
    ax.set_ylabel("ap")
    ax.set_xlabel("create_time")
    #pdf.savefig()
    #plt.clf()
    print ("scatterplot time " + str(datetime.datetime.now() - begin))
    canvas = FigureCanvas(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image =  np.fromstring(s, np.uint8).reshape((height, width, 4))
    return image

def vis_result(exp_name):
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/results/" + exp_name +"/"
    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../data/models/" + exp_name + "/"
    report_dir = os.path.abspath(os.path.dirname(__file__)) + "/../" + "../reports/" + exp_name + "/"
    print(model_dir)
    print(report_dir)
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    result_files = [f for f in os.listdir(data_dir) if re.match(r'^[0-9]*.csv',f)]
    result_files.sort(key=lambda x: int(x.split(".")[0]))
    print(result_files)
    figures = []
    for file in result_files:
        print(file)
        result = pd.read_csv(data_dir + file )
        train_result = pd.read_csv(data_dir+"train_" + file)
        hst = pd.read_csv(data_dir + "history_" + file, header=None)
        
        name = file.split(".")[0]
        #fig = plot(result, train_result, hst, name)
        fig = plot.remote(result = result, train_result = train_result,hst = hst, name = name,sl = 23)
        figures.append({"figure": fig,"name":name})
    
    #pdf = FPDF()
    #for fls in figures:
        #pdf.add_page()
        begin = datetime.datetime.now()
        buf = ray.get(fls["figure"])
        #print (buf)
        matplotlib.image.imsave( report_dir + fls["name"] +".png",buf)
        #pdf.image(report_dir + fls["name"] +".png",0,0,25,26)
        print ("saving time " + str(datetime.datetime.now() - begin))
    #pdf.output(report_dir + exp_name + ".pdf", "F")
begin = datetime.datetime.now()
vis_result("sigmoud_lstm_without_ray")
print("total time taken " + str(datetime.datetime.now() - begin) )
