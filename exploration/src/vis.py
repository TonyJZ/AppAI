import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import gc

from appai_lib import my_io

sns.set(style="dark")
figsize=(16,16)
#in_filename = "../data/stable_all_head_10000.csv"
all_nan = [59, 100, 158, 170, 224, 336, 465, 505, 588]
zero_sum =  [6, 7, 8, 9, 10, 16, 18, 23, 30, 36, 39, 41, 49, 55, 59, 70, 84, 97, 98, 100, 110, 112, 119, 131, 137, 158, 170, 180, 186, 192, 196, 208, 212, 213, 216, 224, 229, 230, 233, 235, 236, 237, 244, 246, 252, 256, 259, 260, 264, 266, 269, 270, 275, 279, 280, 282, 284, 288, 294, 298, 300, 309, 310, 315, 316, 323, 325, 328, 332, 334, 335, 336, 338, 340, 345, 351, 352, 356, 359, 364, 376, 382, 392, 398, 406, 410, 415, 420, 422, 423, 426, 429, 433, 437, 441, 442, 451, 455, 456, 457, 461, 462, 464, 465, 471, 477, 479, 480, 482, 486, 496, 497, 499, 505, 513, 523, 526, 535, 538, 544, 547, 555, 559, 573, 578, 585, 586, 588, 590, 596, 599]
selected_meters = [1, 3, 4, 5, 11, 12, 14, 19, 20, 21, 22, 26, 32, 33, 40, 42, 43, 45, 46, 50, 52, 53, 58, 60,61, 65, 66, 67, 68, 69, 72, 73, 76, 77, 79, 80]
def filter_out(all_data, filter_meter_ids,out_file):
    print(all_data.shape)
    filtered_data = all_data[~all_data.meter_id.isin(filter_meter_ids)]
    print("first filter " + str(filtered_data.shape))

    filtered_data = filtered_data[~((np.isnan(filtered_data.ap))) ]
    print("second filter " + str(filtered_data.shape))

    filtered_data = all_data[all_data.meter_id.isin(selected_meters)]
    print("third filter " + str(filtered_data.shape))

    print ("removed " + str(all_data.shape[0] - filtered_data.shape[0]))
    filtered_data.to_csv(out_file)

def all_ep_graph():
    in_filename = "../data/stable_all_0.csv"
    data_f = pd.read_csv(in_filename, usecols=["meter_data_id","ep","meter_id","create_date"])
    grouped = data_f.groupby('meter_id')

    aggr_data ={"meter_id":[],"max_ep":[],"min_ep":[],"mean_ep":[]}
    with PdfPages('multipage_pdf.pdf') as pdf:
        for name, group in grouped:
            print(name)

            aggr_data["meter_id"].append(name)
            aggr_data["max_ep"].append(group.max()["ep"])
            aggr_data["min_ep"].append(group.min()["ep"])
            aggr_data["mean_ep"].append(group.mean()["ep"])

            fig, ax = plt.subplots(figsize=figsize)

            sns_plot = sns.lineplot(ax = ax, y="ep", x = "create_date", data =group)
            plt.tight_layout()
            pdf.savefig()
            plt.clf()
            plt.close()
            #sns_plot.get_figure().close()
            #sns_plot.get_figure().savefig("output.png")
            gc.collect()

    print(len(data_f["meter_id"].unique()))
    print(data_f.info())
    aggr_df = pd.DataFrame(aggr_data)
    aggr_df.to_csv("../data/stable_aggr.csv")

def find_meter_id_by_index(aggr_filename):
    indexes = [1,4,12,24,40,87,134,140,155,156,162,256,257,334,348,351,352,421,429,506]
    aggr_data = pd.read_csv(aggr_filename)
    res = aggr_data.iloc[indexes,:]
    #print(res)
    return res["meter_id"]

def preprocess(in_file,out_file):
    all_data = load_data(in_file)
    filter_out_ids = all_nan + zero_sum
    print("filtering out " + str(len(filter_out_ids)))
    filter_out(all_data, filter_out_ids, out_file)
    #filtered_data = filter_data(all_data)

def load_data(in_file):
    #column_list = ["meter_data_id","meter_id","pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in","create_date"]
    all_data = pd.read_csv(in_file)
    return all_data

def filter_data(all_data):
    grouped = all_data.groupby('meter_id')
    nan_ids = []
    print("All Nan:")
    for name, group in grouped:
        ep = group["ep"]
        ap = group["ap"]
        #print(create_column)
        #print(type(create_column))
        #print(create_column.size)
        #  count non NaN ep column
        nan_count_ep = len(ep) - ep.count()
        nan_count_ap = len(ap) - ap.count()
        if (nan_count_ep == len(ep)) and (nan_count_ap == len(ap)):
            nan_ids.append(name)
            #print(group)
        #remove all zero columns
    print(", ".join(map(str, nan_ids)))
    print("Zero sums:")
    zero_sum_ids = []
    for name,group in grouped:
        ep = group["ep"]
        ap = group["ap"]
        if(ep.sum() <= 0 and ap.sum() <= 0):
            zero_sum_ids.append(name)
            #print("ep sum" + str(ep.sum()))
            #print("ap sum" + str(ap.sum()))
            #print(group)
    print(", ".join(map(str, zero_sum_ids)))


def draw_detailed_graph(meter_ids, in_file, out_file,all_data = None):
    #if all_data ==None:
    #    all_data = pd.read_csv(in_file)
    column_list = ["pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in"]
    grouped = all_data.groupby('meter_id')
    #plt.tick_params(axis='x', which='both', bottom=False, top=False)

    with PdfPages(out_file) as pdf:
        for name, group in grouped:
            if (meter_ids != None) and (name not in meter_ids):
                continue
            print(name)

            #sorted_group = group.sort_values(by=["create_date"])
            fig = plt.figure(figsize=(40,40))
            fig.suptitle("meter id = " +str(name),fontsize=50)
            oldx = False
            for idx,col in enumerate(column_list):
                if oldx:
                    ax = plt.subplot(4,4,idx+1,sharex=oldx)
                else:
                    ax = plt.subplot(4,4,idx+1)
                oldx = ax
                ax.set_title(col,fontsize=30)
                ax.set_ylabel(col,fontsize=30)
                #ax.set_xlabel("create data",fontsize=30)
                ax.set_xticks([])
                #ax.set_xlabel("create date (April 1 April 25)")
                sns_plot = sns.lineplot(ax = ax, y=col, x = "create_date", data = group)
                # ax.tick_params(axis=u'both', which=u'both',length=0)
                ax.tick_params(axis='y', which='major', labelsize=20)
                # ax.tick_params(axis="x",which="both",labelsize=5)
            #plt.tight_layout()
            pdf.savefig()
            plt.clf()
            plt.close()
            gc.collect()

def read_raw(in_file_first,usecols = None):
    data_list = []
    for i in range (0,4):
        in_file = in_file_first + str(i) + ".csv"
        data = my_io.load_data(in_file, usecols)
        data_list.append(data)
    cat_data = pd.concat(data_list, ignore_index=True)
    return cat_data


if __name__ == "__main__":
    #in_filename = "../data/stable_all_head_10000.csv"
    in_filename = "../data/stable_all_0.csv"
    #in_filename = "../data/filtered_data.csv"
    #meter_ids = find_meter_id_by_index(in_filename)
    #meter_ids = zero_sum + all_nan
    all_data = read_raw("../data/raw/ems_meter_original_data_april_may_",None)

    draw_detailed_graph(selected_meters, None,"filtered_data.pdf",all_data)
    #out_file = "../data/filtered_data_36.csv"
    #preprocess(in_filename,out_file)