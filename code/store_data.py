import mysql.connector
from mysql.connector import Error

def storeDate(accuracy,accuracy_std, precision,precision_std, callback,callback_std, f_score, f_score_std_deviation, metadata_filter, output_mode):
    print("Storing results...")
    try:
        connection = mysql.connector.connect(host='localhost',
                                            database='ebse',
                                            user='root',
                                            password='Root123!')
                                            
        query = "INSERT INTO `results` (`id`, `acc`, `acc_std_deviation`, `prec`, `prec_std_deviation`, `call`, `call_std_deviation`, `fscore`, `fscore_std_deviation`, `filter`, `output_mode`) VALUES (NULL,%s, %s, %s, %s,%s,%s,%s,%s, %s, %s)"
        query_data = (accuracy,accuracy_std, precision,precision_std, callback,callback_std, f_score, f_score_std_deviation, metadata_filter, output_mode)
            
        cursor = connection.cursor()
        cursor.execute(query, query_data)
        connection.commit()
        print("Results stored")

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")