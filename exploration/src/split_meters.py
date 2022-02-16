import os
import pandas as pd

def split_meters(in_file, out_folder):
    df = pd.read_csv(in_file)
    groups = df.groupby('meter_id')
    for name, group in groups:
        group = group.sort_values(by=["create_date"])
        print(name)
        print(group)
        name = out_folder + str(name) + '.csv'
        group.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
        group.to_csv(name, index=False)


if __name__ == "__main__":
    out_dir = "../data/processed/april_may_elapsed_filtered/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    split_meters("../data/processed/april_may_elapsed_filtered.csv", out_dir)