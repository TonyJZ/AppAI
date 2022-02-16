import csv
from datetime import datetime, timedelta
from hdfs import InsecureClient
from multiprocessing import Process
import mysql.connector
import os
import pandas as pd
import pika
import sys
import time

from lib.Config.ConfigUtil import ConfigUtil

from lib.Log.Log import Logger
from lib.Common.Auxiliary import Auxiliary


# Data source: Store Item Demand Forecasting Challenge [train.csv]
# https://www.kaggle.com/c/demand-forecasting-kernels-only/data?login=true

class StoreSimulator:

    # DATE_STRING_FORMAT = '%Y-%m-%d %H:%M:%S'
    DATE_STRING_FORMAT = '%Y-%m-%d'

    START_TIME_SIMULATION = '2013-01-01'
    END_TIME_SIMULATION = '2017-12-31'

    HOURS_PER_DAY = 24
    MINUTES_PER_HOUR = 60
    SECONDS_PER_MINUTE = 60

    MINUTE_START = 1
    SECOND_START = 1

    # customized simulator parameters
    SALES_TIME_INTERVAL_SECONDS = 1#24*3600
    TIME_INTERVAL = 1 #24*3600
    NUMBER_STORES = 10 #250    maximum to 500 stores. hard code in _init_


    # Wait for trigger
    def wait_for_trigger(self):
        date_time = datetime.now()
        while (not ((date_time.minute % self.MINUTE_START == 0) and (date_time.second % self.SECOND_START == 0))):
            time.sleep(0.1)
            date_time = datetime.now()

    # Send task
    # def send_task(self, store_id, record_id):
    def send_task(self, channel, store_id, record_id):
        current_time = datetime.now()
        current_time = datetime(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second)

        if (store_id == self.LIST_STORE_IDS[0]):
            print('Stores\' records for stream %d sent @ %s.' % (self.stream_id, current_time))

        # Build message (JSON string)
        # print('%d in %d' % (store_id, len(self.df_sales_list)))

        record = dict(self.df_sales_list[store_id-1].iloc[record_id])
        records_msg = "{"
        number_items = len(record)

        for i in range (number_items):
            if ((isinstance(record.get(self.raw_headers[i]), datetime)) or (isinstance(record.get(self.raw_headers[i]), str))):
                if (self.raw_headers[i] == 'create_date'):
                    records_msg += '"%s":"%s",' % (self.raw_headers[i], current_time)
                else:
                    records_msg += '"%s":"%s",' % (self.raw_headers[i], record.get(self.raw_headers[i]))
            elif (self.raw_headers[i] == 'store_id'):
                    records_msg += '"store_id":%d,' % (store_id)
            elif (record.get(self.raw_headers[i]) == None):
                records_msg += '"%s":null,' % (self.raw_headers[i])
            else:
                records_msg += '"%s":%s,' % (self.raw_headers[i], record.get(self.raw_headers[i]))

        records_msg = records_msg.strip(',') + "}"

        # Send AMQP messages
        routing_key = 'cnaf-sagw.store.%d' % (store_id) 
        try:
            channel.basic_publish(exchange='amq.topic', routing_key=routing_key, body=records_msg)
            Logger.debug('[Store] Message published for stream_id = %d, store_id = %d.' % (self.stream_id, store_id))
        except:
            Logger.error('[Store] Message could not be published for stream_id = %d, store_id = %d.' % (self.stream_id, store_id))
        # channel.close()

    # Schedule task 
    def schedule_task(self, store_id):
        Logger.debug('[Store] Schedule Task stream_id = %d started with store_id = %d.' % (self.stream_id, store_id))

        record_id = 0
        
        # MQ configuration
        channel = None
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.mq_ip, credentials=self.credentials, heartbeat=0))
            channel = connection.channel()
            Logger.info('[Store] RabbitMQ connection established for store_id = %d.' % store_id)
        except:
            Logger.critical('[Store] RabbitMQ connection could not be established for store_id = %d.' % store_id)

        while True:
            try:
                p = Process(target=self.send_task, args=(channel, store_id, record_id))
                p.start()
                record_id = (record_id + 1) % self.number_records
                time.sleep(self.TIME_INTERVAL)
                Logger.info('[Store] Process in schedule_task was created.')
            except:
                Logger.critical('[Store] Process in schedule_task was not created.')
                # Problem: force application to stop
                break


    # Main function
    def __init__(self, stream_id):

        # Log
        Logger.info('[Store] Simulators module initiated.')

        # Stream
        self.stream_id = stream_id

        # Read config data        
        resources = ConfigUtil.get_value_by_key('resources')
        mq = resources.get('mq')

        # streams = ConfigUtil.get_value_by_key('streams')

        self.LIST_STORE_IDS = list(range(1, self.NUMBER_STORES+1))

        self.streams = ConfigUtil.get_value_by_key('streams')
        self.raw_headers = Auxiliary.get_stream(self.streams, stream_id).get('headers')
        # self.raw_store_ids = Auxiliary.get_stream(self.streams, stream_id).get('type_ids')

        # MQ configuration
        self.mq_ip = mq.get('mq_host_ip')
        self.credentials = pika.credentials.PlainCredentials(mq.get('mq_host_username'), mq.get('mq_host_password'))
        
        # Number of samples
        start_date_test = datetime.strptime(self.START_TIME_SIMULATION, self.DATE_STRING_FORMAT)
        end_date_test = datetime.strptime(self.END_TIME_SIMULATION, self.DATE_STRING_FORMAT)
        date_interval = end_date_test - start_date_test
        number_samples = int((date_interval.days * self.HOURS_PER_DAY * self.MINUTES_PER_HOUR * self.SECONDS_PER_MINUTE + date_interval.seconds) /self.SALES_TIME_INTERVAL_SECONDS)

        # Prepare for test
        print('Simulator %d will run test data from %s to %s.' % (self.stream_id, self.START_TIME_SIMULATION, self.END_TIME_SIMULATION))
        print('Simulator %d will shrink time interval from %d s to %d s.' % (self.stream_id, self.SALES_TIME_INTERVAL_SECONDS, self.TIME_INTERVAL))
        repetition_period = number_samples * self.TIME_INTERVAL
        days_repetition_period = int(time.strftime('%d', time.gmtime(repetition_period))) - 1
        print('Simulator %d has a total number of %d samples (or repetition period of %d s = %dd %s).' % (self.stream_id, number_samples, repetition_period, days_repetition_period, time.strftime('%Hh %Mm %Ss', time.gmtime(repetition_period))))
        print('Simulator %d will run for a total of %d stores.' % (self.stream_id, self.NUMBER_STORES))
        print()  
        
        print('All data has been read from database for stream_id = %d. Ready to start simulation.' % self.stream_id)
        print()  
          
        Logger.info('[Store] Simulator %d will run test data from %s to %s.' % (self.stream_id, self.START_TIME_SIMULATION, self.END_TIME_SIMULATION))
        Logger.info('[Store] Simulator %d will shrink time interval from %d s to %d s.' % (self.stream_id, self.SALES_TIME_INTERVAL_SECONDS, self.TIME_INTERVAL))
        Logger.info('[Store] Simulator %d has a total number of %d samples (or repetition period of %d s = %dd %s).' % (self.stream_id, number_samples, repetition_period, days_repetition_period, time.strftime('%Hh %Mm %Ss', time.gmtime(repetition_period))))
        
        # Read the data from csv
        self.df_sales = pd.read_csv('data/sales_data.csv')

        # Create 500 stores: stores x items (10 x 50), with 1826 records
        self.df_sales['create_date'] = pd.to_datetime(self.df_sales['create_date'])
        self.df_sales['store_id'] = (self.df_sales['store_id']-1)*50+self.df_sales['items']
        self.df_sales = self.df_sales.drop('items', axis=1)

        self.df_sales_list = []
        for store_id in range(1, self.NUMBER_STORES+1):
            df_sales_store = self.df_sales[self.df_sales['store_id']==store_id]
            self.df_sales_list.append(df_sales_store)
        self.number_records = len(self.df_sales_list)
        self.df_sales = None
        
        # Trigger at HH:MM:SS
        self.wait_for_trigger()

        # Create multiple processes (one per store) to start test
        processes = []
    
        for i in range(self.NUMBER_STORES):
            try:
                process = Process(target=self.schedule_task, args=(self.LIST_STORE_IDS[i],))
                process.start()
                processes.append(process)
                Logger.info('[Store] Process %d in __init__ was created.' % i)
            except:
                Logger.critical('[Store] Process %d in __init__ was not created.' % i)