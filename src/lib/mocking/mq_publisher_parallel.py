#!/usr/bin/env python
import datetime
import pika
import time
from multiprocessing import Process


# Wait for trigger
def wait_for_trigger():
    date_time = datetime.datetime.now()
    while (not ((date_time.minute % 1 == 0) and (date_time.second % 10 == 0))):
        time.sleep(0.1)
        date_time = datetime.datetime.now()


def send_task(process_name):

	credentials = pika.credentials.PlainCredentials('guest', 'guest')

	connection = pika.BlockingConnection(pika.ConnectionParameters(host='10.10.10.111', credentials=credentials))
	channel = connection.channel()

	record = '{"create_date": "%s", "store": 0, "items": 0, "sales": 1001.0}' % datetime.datetime.now()

	wait_for_trigger()
	print('Started process %s' % process_name)

	for i in range (100000):
		rk = 'cnaf-sagw.store.%s' % process_name 
		# channel.basic_publish(exchange='amq.topic', routing_key='cnaf-sagw.store.0', body=record)
		channel.basic_publish(exchange='amq.topic', routing_key=rk, body=record)

	print('Published: %s' % record)

	connection.close()



# Create multiple processes (one per store) to start test
number_processes = 2
processes = []

for i in range(1, number_processes+1):
    try:
        process_name = '%d' % i
        process = Process(target=send_task, args=(process_name,))
        process.start()
        processes.append(process)
    except:
        pass
