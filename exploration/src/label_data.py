import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import gc

def load_data(in_file):
    #column_list = ["meter_data_id","meter_id","pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in","create_date"]
    all_data = pd.read_csv(in_file)
    return all_data

def vis(groups):
     with PdfPages('threshold.pdf') as pdf:
         for group in groups:
             outlier = group[group["label"] == 1]
             real =  group[group["label"] == 0]
             #num_subplots = math.ceil(group["elapsed_time"].max() / seg_length)
             #print("num_subplots " + str(num_subplots))
             fig = plt.figure(figsize=(20, 20))
             print(group.meter_id.iloc[0])
             fig.suptitle("meter id = " + str(group.meter_id.iloc[0]),fontsize=30)
             fig.subplots_adjust(hspace = .4)
             ax = plt.subplot(1,1,1)
             sns_plot = sns.scatterplot(ax = ax, y="ap", x ="elapsed_time" , data = real, color = "blue")
             sns_plot = sns.scatterplot(ax = ax, y="ap", x ="elapsed_time" , data = outlier,color = "red")
             ax.legend(labels=["Real","Outlier"])
             pdf.savefig()
             plt.clf()
             plt.close()


def label_data(data):
    grouped = data.groupby('meter_id')
    groups = []
    def f(row,lower,upper):
        if (row.ap < lower or row.ap > upper):
            return 1
        return 0

    for name, group in grouped:
        group = group.sort_values(by=["create_date"])
        mean = group["ap"].mean()
        std = group["ap"].std()
        cut_off = std * 4
        lower,upper = mean - cut_off, mean + cut_off
        group["label"] = group.apply(f,args = (lower,upper), axis=1)
        num_outliers = group.label.sum()
        num_total = group.label.count()
        print("total " + str(num_total) + " outlier " + str(num_outliers) + " ration " + str(num_outliers/num_total) )
        groups.append(group)
    return groups

if __name__ == "__main__":
    data = load_data("../data/filtered_data_36_w_elapsed.csv")
    groupes = label_data(data)
    vis(groupes)