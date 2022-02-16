#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:21:16 2019

@author: Toukir
"""
import pandas as pd
from fbprophet import Prophet
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import gc

from appai_lib import my_io
from appai_lib import preprocessing

def evaluate(y,yhat):
    rmse = np.sqrt(np.mean(y - yhat))
    return rmse


in_file = "../data/processed/april_may_elapsed_filtered_36.csv"
all_data =my_io.load_data(in_file)
sns.set(style="dark")

chinese_holydays = pd.DataFrame(
                {'holiday': 'Tomb-sweeping Day',
                'ds': pd.to_datetime(['2019-04-05']),
                'lower_window': 0,
                'upper_window': 2,}
                )



grouped = all_data.groupby("meter_id")
forcast = None

with PdfPages( "prophet_prediction.pdf") as pdf:
    for name, group in grouped:
        group = group.sort_values(by=["create_date"])
        train_data, test_data = preprocessing.split_data(group,"2019-04-30 00:00:00")
        train_data = train_data.rename(columns={"create_date":"ds","ap":"y"})
        test_data = test_data.rename(columns={"create_date":"ds","ap":"y"})

        p_model = Prophet(changepoint_prior_scale=0.01, changepoint_range=0.95, holidays=chinese_holydays, holidays_prior_scale = 0.05)
        p_model.fit(train_data)

        forcast = p_model.predict(test_data)
        test_data = test_data.reset_index()
        test_data["yhat"] = forcast["yhat"]

        fig = plt.figure(figsize =(25,18))
        fig.suptitle("Meter id = " + str(name))

        ax = plt.subplot(3,1,1)
        ax.set_title("Training data")

        sns.lineplot(ax = ax, y = "y", x = "ds", data = train_data)
        ax.legend(labels = ["Real"])
        every_nth = train_data.y.count()//6
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        ax.set_ylabel("ap")
        ax.set_xlabel("create_time")
        ax = plt.subplot(3,1,2)

        rmse = evaluate(test_data.y, test_data.yhat)
        ax.set_title("RMSE = " + str(rmse))
        #plt.locator_params(axis='x', nticks=10)
        sns.lineplot(ax = ax, y="y", x ="ds" , data = test_data)
        sns.lineplot(ax = ax, y="yhat", x ="ds" , data = test_data)
        error = pd.DataFrame()

        error["ydiff"] = (test_data.yhat - test_data.y).abs()
        error["ds"] = test_data.ds
        sns.lineplot(ax = ax, y="ydiff", x ="ds" , data = error)
        ax.legend(labels = ["Real","Predicted","Difference"])
        every_nth = 700
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        ax.set_ylabel("ap")
        ax.set_xlabel("create_time")
        ax = plt.subplot(3,1,3)

        sns.lineplot(ax = ax, y="yhat", x ="ds" , data = test_data)
        std = test_data.yhat.std()
        ax.fill_between(test_data.ds, test_data.yhat - std*4,test_data.yhat + std*4, color="lightblue",alpha=0.2)

        outside = test_data[((test_data["y"] > test_data.yhat + std *4) | (test_data["y"] < test_data.yhat - std *4))]
        inside = test_data[~((test_data["y"] > test_data.yhat + std *4) | (test_data["y"] < test_data.yhat - std *4))]
        ax.set_title("Normal= " + str(inside.yhat.count()) + " Outlier = " + str(outside.yhat.count()) + " Outlier(%) =" + str(outside.yhat.count() * 100 /test_data.yhat.count()))
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