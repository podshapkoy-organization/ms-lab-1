import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# генерируем выборки: 10000(кал)х1000(разм)
sample_size = 1000
num_samples = 10000

# выбираем распределение Пуассона
samples = np.random.poisson(lam=5.0, size=(sample_size, num_samples))

# вычисляем статистику: выборочное среднее, выборочную дисперсию

# Выборочное среднее
# выбираем по столбцам
sample_means = np.mean(samples, axis=0)
# Гистограмма выборочного среднего
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=100, density=True, color='g', label='Гистограмма')
# Плотность нормального распределения, которое аппроксимирует выборочное среднее (по ЦПТ)
x = np.linspace(sample_means.min(), sample_means.max(), 1000)
plt.plot(x, norm.pdf(x, sample_means.mean(), sample_means.std()), 'r', label='Плотность')
plt.title('Выборочное среднее')
plt.legend()
mean_sample_means = np.mean(sample_means)
var_sample_means = np.var(sample_means)
median_sample_means = np.median(sample_means)
plt.text(5.1, 4,
         f'Ожидание: {mean_sample_means:.2f}', fontsize=12)
plt.text(5.1, 3.7,
         f'Дисперсия: {var_sample_means:.2f}', fontsize=12)
plt.text(5.1, 3.4,
         f'Медиана: {median_sample_means:.2f}',
         fontsize=12)
plt.show()

# выборочная дисперсия
# используем var(), а не variances(), потому что отклонение вдоль оси
sample_variances = np.var(samples, axis=0)

# Гистограмма выборочной дисперсии
plt.figure(figsize=(10, 6))
plt.hist(sample_variances, bins=100, density=True, color='b', label='Гистограмма')

# Плотность хи-квадрат распределения с (n-1) степенями свободы
x = np.linspace(sample_variances.min(), sample_variances.max(), 1000)
plt.plot(x, norm.pdf(x, sample_variances.mean(), sample_variances.std()), 'r', label='Плотность')

plt.title('Выборочная дисперсия')
plt.legend()
mean_sample_variances = np.mean(sample_variances)
var_sample_variances = np.var(sample_variances)
median_sample_variances = np.median(sample_variances)
plt.text(5.5, 1.2,
         f'Ожидание: {mean_sample_variances:.2f}', fontsize=12)
plt.text(5.5, 1.1,
         f'Дисперсия: {var_sample_variances:.2f}', fontsize=12)
plt.text(5.5, 1,
         f'Медиана: {median_sample_variances:.2f}',
         fontsize=12)
plt.show()
