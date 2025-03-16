import pandas as pd
import time
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


df = pd.read_csv('C:\\Users\\cherm\\PycharmProjects\\DataAnal\\lab4\\data\\energy.csv')


def measure_write_time(df, write_func, *args):
    start_time = time.time()
    write_func(df, *args)
    return time.time() - start_time

def measure_read_time(read_func, *args):
    start_time = time.time()
    df = read_func(*args)
    return time.time() - start_time, df


def write_csv(df, filename):
    df.to_csv(filename, index=False)

def read_csv(filename):
    return pd.read_csv(filename)

def write_parquet(df, filename):
    df.to_parquet(filename, index=False)

def read_parquet(filename):
    return pd.read_parquet(filename)


def write_hdf5(df, filename, key):
    df.to_hdf(filename, key=key, mode='w')

def read_hdf5(filename, key):
    return pd.read_hdf(filename, key=key)


def write_sql(df, table_name, db_uri):
    engine = create_engine(db_uri)
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def read_sql(table_name, db_uri):
    engine = create_engine(db_uri)
    return pd.read_sql_table(table_name, engine)


csv_write_time = measure_write_time(df, write_csv, 'data.csv')
csv_read_time, _ = measure_read_time(read_csv, 'data.csv')

parquet_write_time = measure_write_time(df, write_parquet, 'data.parquet')
parquet_read_time, _ = measure_read_time(read_parquet, 'data.parquet')

hdf5_write_time = measure_write_time(df, write_hdf5, 'data.h5', 'table')
hdf5_read_time, _ = measure_read_time(read_hdf5, 'data.h5', 'table')

sql_write_time = measure_write_time(df, write_sql, 'data_table', 'sqlite:///data.db')
sql_read_time, _ = measure_read_time(read_sql, 'data_table', 'sqlite:///data.db')



formats = ['CSV', 'Parquet', 'HDF5', 'SQL']
write_times = [csv_write_time, parquet_write_time, hdf5_write_time, sql_write_time]
read_times = [csv_read_time, parquet_read_time, hdf5_read_time, sql_read_time]

x = range(len(formats))

plt.figure(figsize=(10, 5))
plt.bar(x, write_times, width=0.4, label='Запис', align='center')
plt.bar(x, read_times, width=0.4, label='Читання', align='edge')
plt.xlabel('Формат')
plt.ylabel('Час (секунди)')
plt.xticks(x, formats)
plt.legend()
plt.title('Порівняння швидкості читання та запису для різних форматів')
plt.show()
