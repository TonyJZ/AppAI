#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:37:42 2019

@author: Toukir Imam (toukir@appropolis.com)
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:56:49 2019

@author: Toukir Imam (toukir@appropolis.com)
"""
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta
import dateutil.parser

import os
import re
import matplotlib
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import datetime
sns.set_style("white")
# import ray
def evaluate(y,yhat):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    return rmse


def vis_result(exp_name):
   
    data_dir = os.path.abspath(os.path.dirname(__file__))  + "/../data/results/" + exp_name +"/"
    model_dir = os.path.abspath(os.path.dirname(__file__)) + "/../data/models/" + exp_name + "/"
    report_dir = os.path.abspath(os.path.dirname(__file__))  + "/../reports/"

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    accs = []
    precisions  =[]
    recalls = []
    rmses = []
    result_files = [f for f in os.listdir(data_dir) if re.match(r'^[0-9]*.csv',f)]
    result_files.sort(key=lambda x: int(x.split(".")[0]))
    print(result_files)
    kpp  = []
    for file in result_files:
        print(file)
        result = pd.read_csv(data_dir + file )
        train_result = pd.read_csv(data_dir+"train_" + file)
        hst = pd.read_csv(data_dir + "history_" + file, header=None)
        rmse = evaluate(result.y,result.yhat)
        rmses.append(rmse)
        name = file.split(".")[0]
        
        if "label" in result:
            true_positive = result[(result["label"]  == True) & (result["label_hat"] == True)]
            true_negative = result[(result["label"]  == False) & (result["label_hat"] == False)]
            false_positive = result[(result["label"]  == False) & (result["label_hat"] == True)]
            false_negative = result[(result["label"]  == True) & (result["label_hat"] == False)]
            tp = true_positive.label.count()
            tn = true_negative.label.count()
            fp = false_positive.label.count()
            fn = false_negative.label.count()
            
            acc = (tp +  tn)/(tp + tn + fp + fn)
            if(tp + fp) != 0:
                precision = tp/ (tp + fp)
                precisions.append(precision)
            if (tp + fn ) != 0:
                recall = tp/(tp + fn)
                recalls.append(recall)
            mrg_a = ((tp + fn) * (tp + fp)) / (tp + fn + fp + tn) 
            mrg_b = ((fp + tn) * (fn + tn)) / (tp + fn + fp + tn) 
            expec_agree = (mrg_a + mrg_b) / (tp + fn + fp + tn) 
            obs_agree = (tp + tn) / (tp + fn + fp + tn)     
            cohens_kappa = (obs_agree - expec_agree) / (1 - expec_agree) 
            kpp.append(cohens_kappa)
            print("ck" + str(cohens_kappa))
            accs.append(acc)
            print("acc = " + str(acc))
    mean_rmse = np.mean(rmses)        
    mean_acc = np.mean(acc)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_kpp = np.mean(kpp)
    print(recalls)
    print(" Mean acc " + str(mean_acc))
    print(" Mean precision " + str(mean_precision))
    print(" Mean recall " + str(mean_recall))
    print(" Mean kpp " + str(mean_kpp))
    print(" Mean rmse " + str(mean_rmse))
    
if __name__ == "__main__":
    vis_result("horovod_test")

