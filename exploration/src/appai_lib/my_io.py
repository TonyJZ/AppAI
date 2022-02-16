#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:33:21 2019

@author: toukir
"""
import mysql.connector
import csv
import pandas as pd

def mydb():
        return mysql.connector.connect(host="10.10.10.209", \
                user="appropolis", \
                passwd="appropolis123", \
                database="sagw" \
                )

def row_count(mydb):
	mycursor = mydb.cursor()
	mycursor.execute("SELECT COUNT(*) FROM ems_meter_original_data WHERE create_date >= '2019-04-01' ")
	myresult = mycursor.fetchall()
	return myresult

def save_data(data,out_file):
    data.to_csv(out_file)

def get_meter_table(mydb, out_file_name):
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ems_meter")

    #get the header
    headers = [col[0] for col in mycursor.description]
    myresult = mycursor.fetchall()
    if (len(myresult)> 0):
        filename = out_file_name + ".csv"

        fp = open(filename,'w')
        outfile = csv.writer(fp)
        outfile.writerow(headers)
        outfile.writerows(myresult)
        fp.close()


def to_csv(mydb, chunk_size, out_file_name):
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ems_meter_original_data WHERE create_date >= '2019-04-01' ORDER BY create_date")

    #get the header
    headers = [col[0] for col in mycursor.description]
    file_index = 0
    myresult = mycursor.fetchmany(chunk_size)
    while (len(myresult)> 0):
        filename = out_file_name + "_" + str(file_index) + ".csv"
        print("writing to file " + filename + ": row " + str(file_index * chunk_size) + " to row " + str(file_index* chunk_size + len(myresult)))

        fp = open(filename,'w')
        outfile = csv.writer(fp)
        outfile.writerow(headers)
        outfile.writerows(myresult)
        fp.close()
        myresult = mycursor.fetchmany(chunk_size)
        file_index+=1

def load_data(in_file,column_list = None):
    #column_list = ["meter_data_id","meter_id","pv_a","pv_b","pv_c","pc_a","pc_b","pc_c","ep","pf","hr","hd","ap","rp","uab","ubc","uca","in","create_date"]
    all_data = None
    if column_list == None:
        all_data = pd.read_csv(in_file)
    else:
        all_data = pd.read_csv(in_file,usecols = column_list)

    return all_data