import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
# from __future__ import division
# import warnings
# warnings.filterwarnings("ignore")
import mysql.connector

from lib.Config.ConfigUtil import ConfigUtil

MYSQL_DATABASE_SCHEMA_NAME = 'cnaf'
MYSQL_DATABASE_TABLE_NAME = 'sales_data'

# Read config data
resources = ConfigUtil.get_value_by_key('resources')
mysql_conf = resources.get('mysql')

# Connect to MySQL database
mydb = mysql.connector.connect(
    host=mysql_conf.get('mysql_host_ip'),
    user=mysql_conf.get('mysql_host_username'),
    passwd=mysql_conf.get('mysql_host_password')
)

# Read the data from csv
df_sales = pd.read_csv('data/sales_data.csv')

# Convert date field from string to datetime
df_sales['create_date'] = pd.to_datetime(df_sales['create_date'])

# Add data to database
mycursor = mydb.cursor()

mycursor.execute("SET time_zone = '+08:00'")

for index, row in df_sales.iterrows():
    mysql_cmd = 'INSERT INTO %s.%s(date, store, itens, sales) VALUES(%s, %d, %d, %d);' % (
    MYSQL_DATABASE_SCHEMA_NAME, MYSQL_DATABASE_TABLE_NAME, row.date, row.store, row.itens, row.sales)
    mycursor.execute(mysql_cmd)
    print(index)

# Group all stores x items (10 x 50) per date (daily) = 1826 records
df_sales['create_date'] = pd.to_datetime(df_sales['create_date'])
df_sales = df_sales.groupby('create_date').sales.sum().reset_index()

print(df_sales.shape)

x = df_sales.create_date[0:-1]
y1 = df_sales.sales[0:-1]
y2 = pd.DataFrame(df_sales.sales[1:].reset_index()).sales[:]
dy = y2 - y1

# plot monthly sales
# plt.plot(df_sales.date, df_sales.sales)
plt.plot(x, dy)
plt.show()
