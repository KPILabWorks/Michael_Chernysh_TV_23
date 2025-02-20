import os
import pandas as pd
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

file_paths = [os.path.join(script_dir, "datasets", f) for f in [
    "yellow_tripdata_2015-01.csv",
    "yellow_tripdata_2016-01.csv",
    "yellow_tripdata_2016-02.csv",
    "yellow_tripdata_2016-03.csv"
]]

# Читаємо та об'єднуємо всі файли, одразу перетворюючи `tpep_pickup_datetime` у datetime
df_list = [pd.read_csv(file, nrows=1_000_000, parse_dates=["tpep_pickup_datetime"]) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

print(f"Об'єднаний DataFrame містить {len(df):,} рядків.\n")

# Колонки для тестування
test_columns = {
    "Числові": ["passenger_count", "trip_distance", "fare_amount"],
    "Категоріальні": ["VendorID", "payment_type", "RateCodeID"],
    "Дата/час": ["tpep_pickup_datetime"]
}

# Тестуємо кожну категорію колонок
for category, columns in test_columns.items():
    print(f"🔹 Тестуємо {category} колонки:")

    for col in columns:
        sample_value = df[col].iloc[len(df) // 2]  # Випадкове значення з середини

        # Пошук без індексу
        start_time = time.time()
        result = df[df[col] == sample_value]
        time_no_index = time.time() - start_time

        # Пошук з індексом
        df_indexed = df.set_index(col).sort_index()
        start_time = time.time()
        result_indexed = df_indexed.loc[sample_value]
        time_with_index = time.time() - start_time

        # Вивід результатів
        print(f"  📌 Колонка: {col}")
        print(f"    - Час пошуку без індексу: {time_no_index:.6f} сек.")
        print(f"    - Час пошуку з set_index(): {time_with_index:.6f} сек.")
        print(f"    - Прискорення: {time_no_index / time_with_index:.2f}x\n")

# Column defenitions look here: https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data/code