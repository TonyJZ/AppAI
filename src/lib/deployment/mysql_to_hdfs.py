import json
from datetime import datetime
from typing import List

import mysql.connector

from lib.Config.ConfigUtil import ConfigUtil
from lib.Storage.StorageUtil import StorageUtil

#
# DEFINITIONS
#


MYSQL_DATABASE_SCHEMA_NAME = 'sagw'
MYSQL_DATABASE_TABLE_NAME = 'ems_meter_original_data'

DATE_STRING_FORMAT = '%Y-%m-%d %H:%M:%S'

RECORDS_TIMESTAMPS: List[List[str]] = [
    ['2019-02-01 00:00:00', '2019-03-01 00:00:00'],
    ['2019-03-01 00:00:00', '2019-04-01 00:00:00'],
    ['2019-04-01 00:00:00', '2019-05-01 00:00:00'],
    ['2019-05-01 00:00:00', '2019-06-01 00:00:00'],
    ['2019-06-01 00:00:00', '2019-07-01 00:00:00'],
]

#
# PRECONFIG
#

# get configuration
mysql_config = ConfigUtil.get_value_by_key('resources', 'mysql')
mysql_host_ip = mysql_config.get('mysql_host_ip')
mysql_username = mysql_config.get('mysql_host_username')
mysql_password = mysql_config.get('mysql_host_password')

# Connect to MySQL database
mydb = mysql.connector.connect(
    host=mysql_host_ip,
    user=mysql_username,
    passwd=mysql_password
)

# Get database headers
mycursor = mydb.cursor()


# Create data object from dict format to list format
def create_data_object(content, headers):
    data = []
    for title in headers:
        data.append(content[title] if content[title] is not None else '')
    return data


meter_stream_config = ConfigUtil.get_value_by_key('streams', 'meter')
meter_ids = meter_stream_config.get('type_ids')
raw_headers = meter_stream_config.get('headers')
# For each month
for timestamps in RECORDS_TIMESTAMPS:
    # For each meter id
    for meter_id in meter_ids:
        # Query meter record
        print('Meter ID =', meter_id, ' - Month:', timestamps[0])
        mysql_columns = ''
        for header in raw_headers:
            mysql_columns += ('`%s`, ' % header)
        mysql_columns = mysql_columns.rstrip(', ')

        # mysql_cmd = "SELECT %s FROM %s.%s WHERE (meter_id = %d) AND (create_date = '%s')"
        # % (mysql_columns, MYSQL_DATABASE_SCHEMA_NAME, MYSQL_DATABASE_TABLE_NAME, meter_id, current_timestamp)
        mysql_cmd = "SELECT %s FROM %s.%s WHERE (meter_id = %d) AND (create_date >= '%s') AND (create_date < '%s')" % (
            mysql_columns, MYSQL_DATABASE_SCHEMA_NAME, MYSQL_DATABASE_TABLE_NAME, meter_id, timestamps[0],
            timestamps[1])

        mycursor.execute(" set time_zone = '+08:00'")
        mycursor.execute(mysql_cmd)
        myresults = mycursor.fetchall()

        raw_data = []
        for myresult in myresults:
            # Build message (JSON string)
            records_msg = "{"
            number_items = len(myresult)
            for i in range(number_items):
                if (isinstance(myresult[i], datetime)) or (isinstance(myresult[i], str)):
                    records_msg += '"%s":"%s",' % (raw_headers[i], myresult[i])
                elif myresult[i] == None:
                    records_msg += '"%s":null,' % (raw_headers[i])
                else:
                    records_msg += '"%s":%s,' % (raw_headers[i], myresult[i])
            records_msg = records_msg.strip(',') + "}"

            # Retrieve content (possibly containing None values)
            content = json.loads(records_msg)
            # writing raw data row by row
            StorageUtil.write_raw(meter_stream_config.get('stream'), content['meter_id'],
                                  datetime.strptime(content['create_date'], DATE_STRING_FORMAT), content)

