import mysql.connector
from mysql.connector import Error


accuracy = 1.2
precision = 0.5
callback = 0.25
try:
    connection = mysql.connector.connect(host='localhost',
                                        database='ebse',
                                        user='root',
                                        password='Root123!')
                                        
    query = "INSERT INTO `results` (`id`, `acc`, `prec`, `call`) VALUES (NULL,%s, %s, %s)"
    query_data = (accuracy,precision,callback)
        
    cursor = connection.cursor()
    result = cursor.execute(query, query_data)
    connection.commit()
    print("Stored result {}: \t accuracy={} \t precision={} \t callback= {}".format(result, accuracy, precision, callback))

except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")