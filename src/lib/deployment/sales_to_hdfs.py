from datetime import datetime
import pandas as pd
import json

from lib.Config.ConfigUtil import ConfigUtil
from lib.Storage.StorageUtil import StorageUtil

# Read hdfs config data
hdfs_config = ConfigUtil.get_value_by_key('resources', 'hdfs')

# read data from csv file and convert to pandas DataFrame
df_sales = pd.read_csv('data/sales_data.csv')

# read stream configuration
store_stream_config = ConfigUtil.get_value_by_key('streams', 'store')
# important to make sure the headers in configuration matches with the headers in the csv
raw_headers = store_stream_config.get('headers')
stream = store_stream_config.get('stream')

sales_data_list = json.loads(df_sales.to_json(orient='records'))

for sale_item in sales_data_list:
    sale_date = datetime.strptime(sale_item.get('create_date'), '%Y-%m-%d')
    StorageUtil.write_raw(stream, sale_item.get('store_id'), sale_date, sale_item)

