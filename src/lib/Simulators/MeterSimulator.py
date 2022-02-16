import csv
from datetime import datetime, timedelta
from hdfs import InsecureClient
from multiprocessing import Process
import mysql.connector
import os
import pandas as pd
import pika
import sys
from threading import Thread
import time

from lib.Common.Auxiliary import Auxiliary
from lib.Config.ConfigUtil import ConfigUtil
from lib.Log.Log import Logger

class MeterSimulator:

    MYSQL_DATABASE_SCHEMA_NAME = 'sagw'
    MYSQL_DATABASE_TABLE_NAME = 'ems_meter_original_data'

    DATE_STRING_FORMAT = '%Y-%m-%d %H:%M:%S'

    START_TIME_SIMULATION = '2019-03-03 00:00:00'
    END_TIME_SIMULATION = '2019-08-05 23:59:59'

    HOURS_PER_DAY = 24
    MINUTES_PER_HOUR = 60
    SECONDS_PER_MINUTE = 60

    MINUTE_START = 1
    SECOND_START = 1

    # customized simulator parameters
    METERS_TIME_INTERVAL_SECONDS = 1
    TIME_INTERVAL = 1

    METER_REPLICATION_FACTOR = 10#2000
    # factor < base
    METER_REPLICATION_FACTOR_BASE = 10000


    # Wait for trigger
    def wait_for_trigger(self):
        date_time = datetime.now()
        while (not ((date_time.minute % self.MINUTE_START == 0) and (date_time.second % self.SECOND_START == 0))):
            time.sleep(0.1)
            date_time = datetime.now()

    # Send task
    def send_task(self, meter_id, record_id, channel):
        current_time = datetime.now()
        current_time = datetime(current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute, current_time.second)
        
        # if (meter_id == self.list_meter_ids[0]):
        if (meter_id // self.METER_REPLICATION_FACTOR_BASE == self.list_meter_ids[0]) and (meter_id % self.METER_REPLICATION_FACTOR_BASE == 0):
            print('Meters\' records for stream %d sent @ %s.' % (self.stream_id, current_time))

        # record =  self.meter_data[self.list_meter_ids.index(meter_id)][record_id]
        record =  self.meter_data[self.list_meter_ids.index(meter_id // self.METER_REPLICATION_FACTOR_BASE)][record_id]

        # Build message (JSON string)
        records_msg = "{"
        number_items = len(record)
        for i in range (number_items):
            if ((isinstance(record[i], datetime)) or (isinstance(record[i], str))):
                if (self.raw_headers[i] == 'create_date'):
                    records_msg += '"%s":"%s",' % (self.raw_headers[i], current_time)
                else:
                    records_msg += '"%s":"%s",' % (self.raw_headers[i], record[i])
            elif (self.raw_headers[i] == 'meter_id'):
                records_msg += '"meter_id":%d,' % (meter_id)
            elif (record[i] == None):
                records_msg += '"%s":null,' % (self.raw_headers[i])
            else:
                records_msg += '"%s":%s,' % (self.raw_headers[i], record[i])
        records_msg = records_msg.strip(',') + "}"

        # Send AMQP messages
        routing_key = 'cnaf-sagw.meter.%d' % (meter_id) 
        try:
            channel.basic_publish(exchange='amq.topic', routing_key=routing_key, body=records_msg)
            Logger.debug('[Meter] Message published for stream_id = %d, meter_id = %d.' % (self.stream_id, meter_id))
        except:
            Logger.error('[Meter] Message could not be published for stream_id = %d, meter_id = %d.' % (self.stream_id, meter_id))
        # channel.close()

    # Schedule task 
    def schedule_task(self, meter_id):
        Logger.debug('[Meter] Schedule Task stream_id = %d started with meter_id = %d.' % (self.stream_id, meter_id))

        number_records = len(self.meter_data[self.complete_list_meter_ids.index(meter_id) // self.METER_REPLICATION_FACTOR])
        record_id = 0
        
        # MQ configuration
        channel = None
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.mq_ip, credentials=self.credentials, heartbeat=0))
            channel = connection.channel()
            Logger.info('[Meter] RabbitMQ connection was stablished.')
        except:
            Logger.error('[Meter] RabbitMQ connection could not be stablished.')
        
        while True:
            try:
                # p = Process(target=self.send_task, args=(meter_id, record_id, channel))
                # p.start()
                Thread(target=self.send_task, args=(meter_id, record_id, channel)).start()
                record_id = (record_id + 1) % number_records
                time.sleep(self.TIME_INTERVAL)
                Logger.info('[Meter] Thread in schedule_task was created.')
            except:
                Logger.critical('[Meter] Thread in schedule_task was not created.')
                # Problem: force application to stop
                break


    # Main function
    def __init__(self, stream_id):

        # Log
        Logger.info('[Meter] Simulators module initiated.')

        # Stream
        self.stream_id = stream_id

        # Read config data        
        resources = ConfigUtil.get_value_by_key('resources')
        mysql_conf = resources.get('mysql')
        mq = resources.get('mq')

        streams = ConfigUtil.get_value_by_key('streams')
        self.list_meter_ids = streams.get('meter').get('type_ids')
        self.streams = ConfigUtil.get_value_by_key('streams')
        self.raw_headers = Auxiliary.get_stream(self.streams, stream_id).get('headers')

        # Connect to MySQL database
        try:
            self.mydb = mysql.connector.connect(
                host=mysql_conf.get('mysql_host_ip'),
                user=mysql_conf.get('mysql_host_username'),
                passwd=mysql_conf.get('mysql_host_password')
            )
        except:
            Logger.critical('[Meter] MySQL connection could not be stablished.')

        # Number of samples
        start_date_test = datetime.strptime(self.START_TIME_SIMULATION, self.DATE_STRING_FORMAT)
        end_date_test = datetime.strptime(self.END_TIME_SIMULATION, self.DATE_STRING_FORMAT)
        date_interval = end_date_test - start_date_test
        number_samples = int((date_interval.days * self.HOURS_PER_DAY * self.MINUTES_PER_HOUR * self.SECONDS_PER_MINUTE + date_interval.seconds) /self.METERS_TIME_INTERVAL_SECONDS)

        # Prepare for test
        print('Simulator %d will run test data from %s to %s.' % (self.stream_id, self.START_TIME_SIMULATION, self.END_TIME_SIMULATION))
        print('Simulator %d will run the following meters: %s.' % (self.stream_id, self.list_meter_ids))
        print('Simulator %d will run with a replication factor of: %d.' % (self.stream_id, self.METER_REPLICATION_FACTOR))
        print('Simulator %d will shrink time interval from %d s to %d s.' % (self.stream_id, self.METERS_TIME_INTERVAL_SECONDS, self.TIME_INTERVAL))
        repetition_period = number_samples * self.TIME_INTERVAL
        days_repetition_period = int(time.strftime('%d', time.gmtime(repetition_period))) - 1
        print('Simulator %d has a total number of %d samples (or repetition period of %d s = %dd %s).' % (self.stream_id, number_samples, repetition_period, days_repetition_period, time.strftime('%Hh %Mm %Ss', time.gmtime(repetition_period))))
        print()

        Logger.info('[Meter] Simulator %d will run test data from %s to %s.' % (self.stream_id, self.START_TIME_SIMULATION, self.END_TIME_SIMULATION))
        Logger.info('[Meter] Simulator %d will run the following meters: %s.' % (self.stream_id, self.list_meter_ids))
        Logger.info('Simulator %d will run with a replication factor of: %d.' % (self.stream_id, self.METER_REPLICATION_FACTOR))
        Logger.info('[Meter] Simulator %d will shrink time interval from %d s to %d s.' % (self.stream_id, self.METERS_TIME_INTERVAL_SECONDS, self.TIME_INTERVAL))
        Logger.info('[Meter] Simulator %d has a total number of %d samples (or repetition period of %d s = %dd %s).' % (self.stream_id, number_samples, repetition_period, days_repetition_period, time.strftime('%Hh %Mm %Ss', time.gmtime(repetition_period))))
    
        # Query meter records
        self.meter_data = []
        try:
            mycursor = self.mydb.cursor()
            for meter_id in self.list_meter_ids:
                mysql_columns = ''
                for header in self.raw_headers:
                    mysql_columns += ('`%s`, ' % header)
                mysql_columns = mysql_columns.rstrip(', ')

                mysql_cmd = "SELECT %s FROM %s.%s WHERE (meter_id = %d) AND (create_date >= '%s') AND (create_date <= '%s')" % (mysql_columns, self.MYSQL_DATABASE_SCHEMA_NAME, self.MYSQL_DATABASE_TABLE_NAME, meter_id, self.START_TIME_SIMULATION, self.END_TIME_SIMULATION)
                mycursor.execute("SET time_zone = '+08:00'")
                mycursor.execute(mysql_cmd)

                myresult = mycursor.fetchall()
                self.meter_data.append(myresult)

            print('All data has been read from database for stream_id = %d. Ready to start simulation.' % self.stream_id)
        except:
            Logger.critical('[Meter] Meter query has failed.')
            return

        print()


        # MQ configuration
        self.mq_ip = mq.get('mq_host_ip')
        self.credentials = pika.credentials.PlainCredentials(mq.get('mq_host_username'), mq.get('mq_host_password'))
        

        # Trigger at HH:MM:SS
        self.wait_for_trigger()

        # Create multiple processes (one per meter) to start test
        # processes = []


        self.complete_list_meter_ids = []
        for meter_id in self.list_meter_ids:
            for factor in range(self.METER_REPLICATION_FACTOR):
                # Meter Factor < Meter Base
                self.complete_list_meter_ids.append(self.METER_REPLICATION_FACTOR_BASE*meter_id + factor)

        print(self.list_meter_ids)
        print(self.complete_list_meter_ids)

        # number_threads = len(self.meter_data)
        number_threads = len(self.complete_list_meter_ids)

        for i in range(number_threads):
            # channel = None
            # try:
            #     channel = self.connection.channel()
            #     Logger.info('[Meter] RabbitMQ channel was created for meter_id = %d.' % self.list_meter_ids[i])
            # except:
            #     Logger.critical('[Meter] RabbitMQ channel could not be created for meter_id = %d.' % self.list_meter_ids[i])

            try:
                # process = Process(target=self.schedule_task, args=(self.list_meter_ids[i],channel))
                # process.start()
                thread = Thread(target = self.schedule_task, args=(self.complete_list_meter_ids[i],)).start()
                # processes.append(process)
                Logger.info('[Meter] Thread %d in __init__ was created.' % i)
            except:
                Logger.critical('[Meter] Thread %d in __init__ was not created.' % i)