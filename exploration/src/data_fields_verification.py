import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import dateutil.parser


def first_derivative_ep(df):
    time = df["elapsed_time"]
    ep = df["ep"]
    # df.dtypes
    # print(type(time))
    # print(type(ep))
    # print(time)
    # print(ep)
    dt = np.diff(time.values,1)
    dep = np.diff(ep.values,1)
    d1 = dep / dt * 3600
    print(type(d1))
    print(d1)
    return d1
 
def second_derivative_ep(df):
    time = df["elapsed_time"]
    d1 = df["first_derivative_ep"]
    
    dt = np.diff(time,1)
    dd1 = np.diff(d1,1)
    d2 = dd1 / dt * 3600

    return d2

# calculate power factor by P/S
# where P is the active power, S is the apparent power, Q is the reactive power
# S^2 = P^2 + Q^2
def calculate_pf(df):
    ap = df["ap"].pow(2)
    rp = df["rp"].pow(2)
    s = (ap+rp).pow(1./2)
    pf = df["ap"]/s
    return pf


if __name__ == "__main__":
    input_folder = './data/meters/'
    output_folder = './data/meters/chart/'
    for root, dirs, files in os.walk(input_folder):
        # root: current path
        # dirs: sub-directories
        # files: file names
        for name in files:
            df = pd.read_csv(input_folder + "/" + name)
            first_date = dateutil.parser.parse(df.iloc[0,:]["create_date"])
            df["elapsed_time"] = df.apply(lambda row : (dateutil.parser.parse(row.create_date) - first_date ).total_seconds(), axis =1)

            pf = calculate_pf(df)
            df["pf_cal"] = pf

            df["pf_diff"] = df["pf_cal"] - df["pf"] 

            d1 = first_derivative_ep(df)
            d1list = d1.tolist()
            d1list.insert(0,0)
            # print(d1list)
            # print(type(d1))
            # np.insert(d1, 0, values = [0])
            # print(d1)
            # print(df.count())
            print(len(d1list))
            df["first_derivative_ep"] = pd.Series(d1list, index=df.index)

            df["ap_diff"] = df["first_derivative_ep"] - df["ap"]

            d2 = second_derivative_ep(df)
            d2list = d2.tolist()
            d2list.insert(0,0)
            print(len(d2list))
            df["second_derivative_ep"] = pd.Series(d2list, index=df.index)

            seg_length = timedelta(days=26).total_seconds()
            with PdfPages(output_folder + "/" + name + ".pdf") as pdf:
                # num_subplots = math.ceil(group["elapsed_time"].max() / seg_length)
                fig = plt.figure(figsize =(40,40))

                fig.suptitle("meter id = " + name,fontsize=30)
                ax1 = plt.subplot(7,1,1)
                sns_plot = sns.lineplot(ax = ax1, y="ep", x = "elapsed_time", data = df)
                ax1.set_title("ep")
                
                ax2 = plt.subplot(7,1,2)
                ax2.set_title("first_derivative_ep")
                sns_plot = sns.lineplot(ax = ax2, y="first_derivative_ep", x = "elapsed_time", data = df)
                
                ax3 = plt.subplot(7,1,3)
                ax3.set_title("ap")
                sns_plot = sns.lineplot(ax = ax3, y="ap", x = "elapsed_time", data = df)

                ax4 = plt.subplot(7,1,4)
                ax4.set_title("ap_diff")
                sns_plot = sns.lineplot(ax = ax4, y="ap_diff", x = "elapsed_time", data = df)

                ax5 = plt.subplot(7,1,5)
                ax5.set_title("pf")
                sns_plot = sns.lineplot(ax = ax5, y="pf", x = "elapsed_time", data = df)

                ax6 = plt.subplot(7,1,6)
                ax6.set_title("pf_cal")
                sns_plot = sns.lineplot(ax = ax6, y="pf_cal", x = "elapsed_time", data = df)

                ax7 = plt.subplot(7,1,7)
                ax7.set_title("pf_diff")
                sns_plot = sns.lineplot(ax = ax7, y="pf_diff", x = "elapsed_time", data = df)
                # sm.graphics.tsa.plot_pacf(group["ap"],ax= ax3,lags=500)
                pdf.savefig()
                plt.clf()
                plt.close()



