import pandas as pd
from pandas import Series
import math
import os
from statsmodels.tsa.stattools import adfuller


def calculate_statistics(arr, segNum):
    # print(arr.mean())
    # print(arr.iloc[0])
    # print(arr.iloc[1:10])
    meanList = []
    varList = []
    split = math.floor(len(arr) / segNum)
    for i in range(0, segNum):
        if i+1 < segNum :
            X = arr.iloc[i*split: (i+1)*split]
            mean = X.mean()
            var = X.var()
            meanList.append(mean)
            varList.append(var)
        else :
            pass
    return (meanList, varList)        

def ADF_test(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    return result

if __name__ == "__main__":
    input_folder = './data/meters/'
    output_folder = './data/meters/chart/'
    for root, dirs, files in os.walk(input_folder):
        # root: current path
        # dirs: sub-directories
        # files: file names
        for name in files:
            df = pd.read_csv(input_folder + name)
            meanList, varList = calculate_statistics(df["ap"], 3)
            result = ADF_test(df["ap"])
            
            file=open(output_folder + name + ".txt",'w')
            file.write("mean\n")
            file.write(str(meanList))
            file.write("\nvar\n")
            file.write(str(varList))

            file.write("\nADF test\n")
            file.write('p-value: %f' % result[1])
            file.close()