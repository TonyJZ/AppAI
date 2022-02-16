from datetime import datetime as datetime
from datetime import timedelta

from math import ceil, floor
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import re
import dateutil.parser

import sys

from storage.storage_util import *


NUMBER_TICKS = 4


def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse

# Get meter_id, year and month as an argument
if (len(sys.argv) != 4):
    print('Usage:  python %s <meter_id> <YYYY> <MM>' % sys.argv[0])
    exit(1)

meter_id = int(sys.argv[1])
year = int(sys.argv[2])
month = int(sys.argv[3])
year_month = '%4d-%02d' % (year, month)
date = datetime.strptime(year_month, '%Y-%m')

# Retrieve results from HDFS
training_results = read_training_results(meter_id, 0)
test_results = read_result(meter_id, date, 0)

# Retrieve fields
training_data = pd.DataFrame()
for i in range (len(TRAINING_RESULTS_HEADERS)):
    training_data[TRAINING_RESULTS_HEADERS[i]] = training_results.iloc[:,i]

result_data = pd.DataFrame()


print(TRAINING_RESULTS_HEADERS)
print(training_data.shape)



for i in range (len(RESULTS_HEADERS)):
    result_data[RESULTS_HEADERS[i]] = test_results.iloc[:,i]
# result_data['timestamp'] = test_results.iloc[:,0]
# result_data['x_n'] = test_results.iloc[:,1]
# for i in range (1,P_STEPS+1):
#     field = 'yhat_' + str(i)
#     result_data[field] = test_results.iloc[:,1+i]
n_samples = len(result_data.x_n)

# Definitions
list_rmse = []

filename = str(meter_id) + '.pdf'

plt.rcParams.update({'font.size': 22})

# One file per meter
with PdfPages(filename) as pdf:
    # fig = plt.figure(figsize =(25,40))
    fig = plt.figure(figsize =(35,60))
    fig.suptitle("Results for Meter ID = " + str(meter_id))

    # Plot main page (training data)
    ax = plt.subplot(3, 1, 1)
        
    plt.plot(training_data.timestamp, training_data.y, color = '#0439CC', linewidth=3)
    plt.plot(training_data.timestamp, training_data.yhat, color = '#20E5FF', linewidth=3)
    plt.plot(training_data.timestamp, training_data.ydiff, color = '#FF822B', linewidth=3)    

    ticks_pos = []
    for i in range (NUMBER_TICKS):
        ticks_pos.append(i*(floor((len(training_data.timestamp)-1)/(NUMBER_TICKS-1))))

    print(ticks_pos)
    for tick in ticks_pos:
        print(training_data.timestamp[tick])

    rmse_train = evaluate( training_data.y,  training_data.yhat)
    plt.legend(["Real (y)", "Single-Step Prediction ($\\hat{y}_1$)", "Prediction Error (${y}_{diff}$)"])
    plt.xticks(ticks_pos)
    ax.set_title("Training data: RMSE = %f" % rmse_train)
    ax.set_ylabel('y, $\\hat{y}_1$, ${y}_{diff}$') 
    ax.set_xlabel("timestamp")

    # Plot test data
    ax = plt.subplot(3, 1, 2)
    plt.plot(result_data.timestamp, result_data.x_n, linewidth=3, label = "Real")
    yhat_values = result_data['yhat_1'].iloc[0:n_samples-1]

    plt.plot(result_data.timestamp.iloc[1:], yhat_values, linewidth=3, label = "Single-Step Prediction")

    number_samples = len(result_data.timestamp)-1

    ticks_pos = []
    for i in range (NUMBER_TICKS):
        ticks_pos.append(i*floor(number_samples/(NUMBER_TICKS-1)))

    yhat_values_sections = []
    number_sections = ceil(number_samples / P_STEPS)
    range_prediction_levels = range (1, P_STEPS+1)

    for section in range(number_sections):
        if ((section != number_sections-1) or (number_samples % P_STEPS == 0)):
            for prediction_level in range_prediction_levels:
                yhat_label = 'yhat_%d' % prediction_level
                y_value = float(result_data[yhat_label].iloc[section * P_STEPS + 1])
                yhat_values_sections.append(y_value)
        else:
            for prediction_level in range(1, number_samples % P_STEPS + 1):
                yhat_label = 'yhat_%d' % prediction_level
                y_value = float(result_data[yhat_label].iloc[section * P_STEPS + 1])
                yhat_values_sections.append(y_value)

    print('*** ', len(result_data.timestamp.iloc[1:]), len(yhat_values_sections))
    plt.plot(result_data.timestamp.iloc[1:], yhat_values_sections, linewidth=3, label = "Multi-step Prediction")

    plt.legend(["Real (y)", "Single-Step Prediction ($\\hat{y}_1$)", "Multi-Step Prediction ($\\hat{y}_{%d}$)" % P_STEPS])
    plt.xticks(ticks_pos)
    ax.set_title("Test data")
    ax.set_ylabel('y, $\\hat{y}_1$, $\\hat{y}_{%d}$' % P_STEPS) # '$\\hat{y}_1$'
    ax.set_xlabel("timestamp")

    # RMSEs plot
    for prediction_level in range_prediction_levels:
        yhat_label = 'yhat_%d' % prediction_level
        y_values = result_data.x_n.iloc[0:n_samples-prediction_level]
        yhat_values = result_data[yhat_label].iloc[0:n_samples-prediction_level]
        rmse = evaluate(y_values, yhat_values)
        list_rmse.append(rmse)

    ax = plt.subplot(3, 1, 3)
    plt.plot(range_prediction_levels, list_rmse, 'ko--', linewidth=3, markersize=12)
    plt.grid()
    ax.set_title("RMSE per prediction level")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("prediction level [discrete]")

    pdf.savefig()
    plt.clf()

    # Config for plots of all prediction levels
    plots_per_page = 5
    page_initial_indeces = []
    for i in range(ceil(P_STEPS / plots_per_page)):
        page_initial_indeces.append(i * plots_per_page)

    # For each page of the pdf file
    for page_index in page_initial_indeces:
        # Vary on each prediction level
        for prediction_level in range (1 + page_index, 1 + min(page_index + plots_per_page, P_STEPS)):
            ax = plt.subplot(plots_per_page, 1, 1 + (prediction_level - 1) % plots_per_page)
            
            plt.plot(result_data.timestamp, result_data.x_n, label = "Real (y)")
            yhat_label = 'yhat_%d' % prediction_level

            y_values = result_data.x_n.iloc[0:n_samples-prediction_level]
            yhat_values = result_data[yhat_label].iloc[0:n_samples-prediction_level]
            rmse = evaluate(y_values, yhat_values)
            list_rmse.append(rmse)

            plt.plot(result_data.timestamp.iloc[prediction_level:], yhat_values, label = "Predicted $\\hat{y}_{%d}$" % prediction_level)
            plt.xticks(ticks_pos)
            plt.legend(["Real (y)", "Predicted $\\hat{y}_{%d}$" % prediction_level])
            ax.set_title("Training data for %d$^{th}$ prediction: RMSE = %s" % (prediction_level, str(rmse)))
            ax.set_ylabel("y, $\\hat{y}_{%d}$" % prediction_level)
            ax.set_xlabel("timestamp")
        
        pdf.savefig()
        plt.clf()

    # plt.clf()
    plt.close()
