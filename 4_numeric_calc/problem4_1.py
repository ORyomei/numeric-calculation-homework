# from .numeric_calculation import NumericCalculation
from numeric_calculation import NumericCalculation
import datetime
import numpy as np
import matplotlib.pyplot as plt


def a(x: np.ndarray, v: np.ndarray, epsilon: np.float64) -> np.ndarray:
    _a = 2.0 + epsilon * x[0] * x[0] - x[0]
    return np.array([_a], dtype=np.float64)


figure = plt.figure()
axes = figure.add_subplot(111)
axes.set_aspect('equal', adjustable='box')

epsilons = [0.0, 0.001, 0.01, 0.1]
colors = ["b", "g", "r", "c"]
for epsilon, color in zip(epsilons, colors):
    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=np.array([1], dtype=np.float64),
                               vInitial=np.array([0], dtype=np.float64),
                               tDelta=4 * np.pi / 50000,
                               tInitial=0.0,
                               tFinal=4 * np.pi)
    calculator.setDataFileName("data_{}.csv".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')))
    calculator.setMethod("Symplectic")

    def _a(x, v):
        return a(x, v, epsilon)

    calculator.setEquation(_a)
    calculator.start(logEvery=50)
    calculator.plot(axes, color, label="$\epsilon = {}$".format(epsilon))

axes.legend()
# figure.savefig("plot4_1.png")
plt.show()
