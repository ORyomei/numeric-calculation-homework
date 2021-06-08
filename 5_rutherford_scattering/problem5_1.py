# from .numeric_calculation import NumericCalculation
from numeric_calculation import NumericCalculation
import numpy as np
import datetime
import matplotlib.pyplot as plt
from os import makedirs


def a(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return x / np.power(np.linalg.norm(x, ord=2), 3)


folderName = "data_{}".format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))
makedirs(folderName)

figure = plt.figure()
axes = figure.add_subplot(111)
axes.set_aspect('equal', adjustable='box')
axes.set_xlim([-0.5, 0.5])
axes.set_ylim([-0.5, 0.5])

colors = ["b", "g", "r", "c", "m", "y", "k"] * 3
y_0s = [0.1 - 0.2 * i / 19 for i in range(20)]

# y_0s = [0.1, -0.1]

for y_0, color in zip(y_0s, colors):
    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=np.array([-10, y_0], dtype=np.float64),
                               vInitial=np.array([10, 0], dtype=np.float64),
                               tDelta=3.0 / 50000,
                               tInitial=0.0,
                               tFinal=3.0)
    calculator.setDataFileName("{}/data_{}.csv".format(folderName,
                                                       datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')))
    calculator.setMethod("Symplectic")
    calculator.setEquation(a)
    calculator.start(logEvery=5)
    calculator.plot(axes, color, label="$y_0 = {}$".format(round(y_0, 3)))

# axes.legend()
# figure.savefig("plot5_1.png")
plt.show()
