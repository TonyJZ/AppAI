import mysql.connector
import csv
import os

mydb = mysql.connector.connect(
  host="10.10.10.209",
  user="appropolis",
  passwd="appropolis123",
  database="sagw"
)

def time_zone_test(mydb):
    mycursor = mydb.cursor()
    mycursor.execute(" set time_zone = '+08:00'")
    mycursor.execute("select create_date from ems_meter_original_data limit 1;")
    myresult = mycursor.fetchall()
    print(myresult)
    
def row_count(mydb):
    mycursor = mydb.cursor()
    #mycursor.execute("SELECT COUNT(*) FROM ems_meter_original_data WHERE create_date >= '2019-04-01' ORDER BY create_date")
    mycursor.execute("SELECT COUNT(*) FROM ems_meter_original_data")
    
    myresult = mycursor.fetchall()
    print(myresult)

def to_csv(mydb, chunk_size, out_file_name):
    mycursor = mydb.cursor()

    #mycursor.execute("SELECT * FROM ems_meter_original_data LIMIT 5")

    #all
    #seet shanghai timezone
    mycursor.execute("set time_zone = '+08:00'")
    mycursor.execute("SELECT * FROM ems_meter_original_data WHERE create_date >= '2019-04-01' ORDER BY create_date ASC")

    #stable
    #mycursor.execute("SELECT * FROM ems_meter_original_data WHERE create_date >= '2019-04-01' ORDER BY create_date")

    #get the header
    headers = [col[0] for col in mycursor.description]
    file_index = 0
    myresult = mycursor.fetchmany(chunk_size)
    while (len(myresult)> 0) and file_index < 50:
        filename = out_file_name + str(file_index) + ".csv"
        print("writing to file " + filename + ": row " + str(file_index * chunk_size) + " to row " + str(file_index* chunk_size + len(myresult)))
        fp = open(filename,'w')
        outfile = csv.writer(fp)
        outfile.writerow(headers)
        outfile.writerows(myresult)
        fp.close()
        myresult = mycursor.fetchmany(chunk_size)
        file_index+=1
        print(len(myresult))

if __name__ == "__main__":
    #time_zone_test(mydb)
    #row_count(mydb)
    out_dir = "../data/raw/july/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    to_csv(mydb,1000000, out_dir)
