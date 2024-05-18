import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")

# Определяем вид, который встречается много и мало раз
most_common_species = data['Species'].mode()[0]
least_common_species = data['Species'].value_counts().idxmin()

setosa_data = data[data['Species'] == 'setosa']
versicolor_data = data[data['Species'] == 'versicolor']
virginica_data = data[data['Species'] == 'virginica']
# Расчет выборочного среднего, выборочной дисперсии, выборочной медианы
setosa_total_area = setosa_data['Sepal.Length'] * setosa_data['Sepal.Width'] + setosa_data['Petal.Length'] * \
                    setosa_data['Petal.Width']
versicolor_total_area = versicolor_data['Sepal.Length'] * versicolor_data['Sepal.Width'] + versicolor_data[
    'Petal.Length'] * versicolor_data['Petal.Width']
virginica_total_area = virginica_data['Sepal.Length'] * virginica_data['Sepal.Width'] + virginica_data['Petal.Length'] * \
                       virginica_data['Petal.Width']

sort_setosa = np.sort(setosa_total_area)
sort_versicolor = np.sort(versicolor_total_area)
sort_virginica = np.sort(virginica_total_area)

print("Частый вид:", most_common_species)
print("Нечастный вид:", least_common_species)

# Расчет выборочного среднего, выборочной дисперсии, выборочной медианы
total_area = data['Sepal.Length'] * data['Sepal.Width'] + data['Petal.Length'] * data['Petal.Width']
total_area_mean = total_area.mean()
total_area_var = total_area.var()
total_area_median = total_area.median()

print("Выборочное среднее суммарной площади:", total_area_mean)
print("Выборочная дисперсия суммарной площади:", total_area_var)
print("Выборочная медиана суммарной площади:", total_area_median)

species_stats = data.groupby('Species')[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].agg(
    {'Sepal.Length': ['mean', 'var', 'median', lambda x: np.percentile(x, 40)],
     'Sepal.Width': ['mean', 'var', 'median', lambda x: np.percentile(x, 40)],
     'Petal.Length': ['mean', 'var', 'median', lambda x: np.percentile(x, 40)],
     'Petal.Width': ['mean', 'var', 'median', lambda x: np.percentile(x, 40)]})

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(total_area, bins=20, density=True, color='b', label='Total Area')
plt.title('Гистограмма суммарной площади')
plt.xlabel('Суммарная площадь')
plt.ylabel('Плотность вероятности')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(total_area)
plt.title('Box-plot суммарной площади')
plt.ylabel('Суммарная площадь')
plt.tight_layout()
plt.show()


def empirical_cdf(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y = np.arange(1, n + 1) / n
    return sorted_data, y


plt.figure(figsize=(18, 6))

# Setosa
plt.subplot(1, 3, 1)
plt.hist(setosa_total_area, bins=20, density=True, color='r', label='Setosa')
plt.title('Гистограмма суммарной площади для Setosa')
plt.xlabel('Суммарная площадь')
plt.ylabel('Плотность вероятности')
plt.legend()

plt.twinx()
setosa_total_area_sorted, setosa_ecdf = empirical_cdf(setosa_total_area)
plt.plot(setosa_total_area_sorted, setosa_ecdf, color='b', label='Empirical CDF')
plt.ylabel('Эмпирическая функция распределения')
plt.legend(loc='upper right')

plt.subplot(1, 3, 2)
plt.hist(versicolor_total_area, bins=20, density=True, color='g', label='Versicolor')
plt.title('Гистограмма суммарной площади для Versicolor')
plt.xlabel('Суммарная площадь')
plt.ylabel('Плотность вероятности')
plt.legend()

plt.twinx()
versicolor_total_area_sorted, versicolor_ecdf = empirical_cdf(versicolor_total_area)
plt.plot(versicolor_total_area_sorted, versicolor_ecdf, color='b', label='Empirical CDF')
plt.ylabel('Эмпирическая функция распределения')
plt.legend(loc='upper right')

plt.subplot(1, 3, 3)
plt.hist(virginica_total_area, bins=20, density=True, color='b', label='Virginica')
plt.title('Гистограмма суммарной площади для Virginica')
plt.xlabel('Суммарная площадь')
plt.ylabel('Плотность вероятности')
plt.legend()

plt.twinx()
virginica_total_area_sorted, virginica_ecdf = empirical_cdf(virginica_total_area)
plt.plot(virginica_total_area_sorted, virginica_ecdf, color='r', label='Empirical CDF')
plt.ylabel('Эмпирическая функция распределения')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.boxplot(setosa_total_area)
plt.title('Box-plot суммарной площади для Setosa')
plt.ylabel('Суммарная площадь')

plt.subplot(1, 3, 2)
plt.boxplot(versicolor_total_area)
plt.title('Box-plot суммарной площади для Versicolor')
plt.ylabel('Суммарная площадь')

plt.subplot(1, 3, 3)
plt.boxplot(virginica_total_area)
plt.title('Box-plot суммарной площади для Virginica')
plt.ylabel('Суммарная площадь')

plt.tight_layout()
plt.show()

