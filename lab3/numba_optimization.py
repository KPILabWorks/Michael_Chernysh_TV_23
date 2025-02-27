import numpy as np
import time
import matplotlib.pyplot as plt
from numba import jit



# Чиста реалізація Python
def energy_needs_python(power, temp, factor):
    result = []
    for p, t, f in zip(power, temp, factor):
        result.append(p * (1 + 0.01 * (t - 25)) * f)
    return result


# Оптимізована версія з Numba
@jit(nopython=True)
def energy_needs_numba(power, temp, factor):
    result = np.empty_like(power)
    for i in range(len(power)):
        result[i] = power[i] * (1 + 0.01 * (temp[i] - 25)) * factor[i]
    return result


# Генерація тестових даних
sizes = [10 ** i for i in range(1, 7)]  # Різні розміри вхідних даних
python_times = []
numba_times = []

for size in sizes:
    power = np.random.uniform(100, 500, size)
    temp = np.random.uniform(15, 35, size)
    factor = np.random.uniform(0.8, 1.2, size)

    # Час виконання Python-версії
    start = time.time()
    energy_needs_python(power, temp, factor)
    python_times.append(time.time() - start)

    # Час виконання Numba-версії
    start = time.time()
    energy_needs_numba(power, temp, factor)
    numba_times.append(time.time() - start)

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(sizes, python_times, label='Python', marker='o')
plt.plot(sizes, numba_times, label='Numba', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Розмір вхідних даних')
plt.ylabel('Час виконання (секунди)')
plt.title('Порівняння продуктивності Python vs Numba')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
