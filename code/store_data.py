import mysql.connector
from mysql.connector import Error

def storeDate(accuracy, precision, callback, f_score, metadata_filter):
    print("Storing result: \t filter={} \t accuracy={} \t precision={} \t callback= {} \t f-score= {}".format(metadata_filter, accuracy, precision, callback, f_score))
    try:
        connection = mysql.connector.connect(host='localhost',
                                            database='ebse',
                                            user='root',
                                            password='Root123!')
                                            
        query = "INSERT INTO `results` (`id`, `acc`, `prec`, `call`, `fscore`, filter) VALUES (NULL,%s, %s, %s, %s, %s)"
        query_data = (accuracy,precision,callback, f_score, metadata_filter)
            
        cursor = connection.cursor()
        cursor.execute(query, query_data)
        connection.commit()
        print("Stored result:\t filter={} \t accuracy={} \t precision={} \t callback= {} \t f-score= {}".format(metadata_filter, accuracy, precision, callback, f_score))

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")