
# from dotCSV
# https://www.youtube.com/watch?v=w2RJ1D6kz-o&list=PL-Ogd76BhmcCO4VeOlIH93BMT5A_kKAXp
# IA NOTEBOOK #1 | Regresión Lineal y Mínimos Cuadrados Ordinarios | Programando IA

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)

X = np.array(boston.data[:, 5])
Y = np.array(boston.target)

plt.scatter(X, Y, alpha=0.3)
plt.show()

# add column of 1s for independent term
X = np.array([np.ones(506) ,X]).T

# @is used to do matrix multiplication
B = np.linalg.inv(X.T @ X) @ X.T @ Y

plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c="red")
plt.show()