import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Завантаження даних
df = pd.read_csv('C:\\Users\\cherm\\PycharmProjects\\DataAnal\\lab6\\data\\dataset.csv')

# Об'єднання дати та часу в єдиний datetime
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
print(f"Кількість некоректних datetime: {df['datetime'].isnull().sum()}")
df = df.dropna(subset=['datetime'])  # Видалення рядків з некоректними датами

# Створення стовпця з годиною
df['hour'] = df['datetime'].dt.hour

# Перевірка на наявність пропущених значень
print("Пропущені значення в кожному стовпці:")
print(df.isnull().sum())

# Перевірка унікальних значень в стовпці 'activity'
print("Унікальні значення активностей:")
print(df['activity'].unique())

# Перевірка на наявність необхідних стовпців
required_columns = ['acceleration_x', 'acceleration_y', 'acceleration_z', 'activity']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Відсутні стовпці: {', '.join(missing_columns)}")
else:
    print("Всі необхідні стовпці присутні.")

# Обчислення модуля прискорення
df['acc_magnitude'] = np.sqrt(df['acceleration_x'] ** 2 + df['acceleration_y'] ** 2 + df['acceleration_z'] ** 2)

# Налаштування фільтру
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Параметри фільтру
cutoff = 3  # Частота зрізу
fs = 50  # Частота дискретизації
order = 4  # Порядок фільтру

# Застосування фільтру
df['acc_filtered'] = butter_lowpass_filter(df['acc_magnitude'], cutoff, fs, order)

# Визначення середнього прискорення для кожної активності
activities = [0, 1]
for activity in activities:
    subset = df[df['activity'] == activity]
    if subset.empty:
        print(f"Немає даних для активності: {activity}")
        continue

    avg_acc = subset['acc_filtered'].mean()
    print(f"Середнє прискорення для {activity}: {avg_acc:.3f} м/с²")

    plt.figure(figsize=(12, 6))
    plt.plot(subset['acc_filtered'].values, label=f'Фільтроване прискорення ({activity})', color='r')
    plt.title(f'Фільтроване прискорення для {activity}')
    plt.xlabel('Час (зразки)')
    plt.ylabel('Прискорення (м/с²)')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

# Візуалізація фільтрованого прискорення для всіх активностей
plt.figure(figsize=(12, 6))
for activity in activities:
    subset = df[df['activity'] == activity]
    if not subset.empty:
        plt.plot(subset['acc_filtered'].values, label=f'Активність {activity}')
plt.title('Фільтроване прискорення для різних активностей')
plt.xlabel('Час (зразки)')
plt.ylabel('Прискорення (м/с²)')
plt.legend()
plt.grid(True)
plt.show()

# Перевірка на аномалії
threshold = 10
anomalies = df[df['acc_magnitude'] > threshold]
print(f"Кількість аномальних значень: {len(anomalies)}")

# Статистики для кожної активності
for activity in activities:
    subset = df[df['activity'] == activity]
    if subset.empty:
        print(f"Немає даних для активності: {activity}")
        continue

    avg_acc = subset['acc_filtered'].mean()
    std_acc = subset['acc_filtered'].std()
    min_acc = subset['acc_filtered'].min()
    max_acc = subset['acc_filtered'].max()

    print(f"Статистики для активності {activity}:")
    print(f"Середнє: {avg_acc:.3f} м/с²")
    print(f"Стандартне відхилення: {std_acc:.3f} м/с²")
    print(f"Мінімум: {min_acc:.3f} м/с²")
    print(f"Максимум: {max_acc:.3f} м/с²")
    print("-" * 40)

# Моделювання з використанням Random Forest
X = df[['acceleration_x', 'acceleration_y', 'acceleration_z']]
y = df['activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")

# Перевірка правильності обробки годин
print("Унікальні години в даних:")
print(df['hour'].unique())

# Аггрегація середнього прискорення за годинами
avg_acc_by_hour = df.groupby('hour')['acc_filtered'].mean()
print("Середнє прискорення за годинами:")
print(avg_acc_by_hour)

plt.figure(figsize=(10, 6))
avg_acc_by_hour.plot(kind='line', marker='o', color='b')
plt.title('Середнє фільтроване прискорення за годинами дня')
plt.xlabel('Година')
plt.ylabel('Середнє прискорення (м/с²)')
plt.grid(True)
plt.show()
