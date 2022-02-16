import os
import sys
import csv
import pandas as pd
from datetime import datetime
from datetime import timedelta


if __name__ == "__main__":
    cur_path = os.path.dirname(__file__)
    input_folder = cur_path + "/../data/processed/june"
    output_folder = cur_path + "/../data/processed/june_tagAnomaly"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    dataVolumn = "create_date"
    valueVolumn = "ep_diff"
    idxVolumn = "meter_data_id"

    for root, dirs, files in os.walk(input_folder):
    # root: current path
    # dirs: sub-directories
    # files: file names
        for name in files:
            df = pd.read_csv(input_folder + "/" + name)
            df.set_index(idxVolumn)
            df["value"] = df[valueVolumn]
            df["date"] = pd.to_datetime(df[dataVolumn], format="%Y-%m-%d %H:%M:%S")

            df.to_csv(output_folder + "/" + name, index=False)
