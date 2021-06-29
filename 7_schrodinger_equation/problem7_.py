# from .numeric_calculation import NumericCalculation
from numeric_calculation import NumericCalculation
import numpy as np
import matplotlib.pyplot as plt
import datetime
from os import makedirs
from scipy.optimize import fsolve
from random import choice


def V(t: np.float64):
    return -3.0 / np.power(np.cosh(t), 2)


def a(x: np.ndarray, t: np.float64, eigenValue: np.float64) -> np.ndarray:
    return (V(t) - eigenValue) * x


figure = plt.figure(figsize=(16, 7))
axes = figure.add_subplot(111)

for i in range(5000):
    e = -0.093 + 0.000001 * i
    initialValues = (np.array([10**-5], dtype=np.float64),
                     np.array([10**-5 * np.sqrt(V(-20) - e)],
                              dtype=np.float64), 20 / 500, -20, 0)
    calculator = NumericCalculation()
    calculator.setInitialValue(*initialValues)

    def _a(x: np.ndarray, v: np.ndarray, t: np.float64) -> np.ndarray:
        return a(x, t, e)

    calculator.setEquation(_a)
    calculator.setMethod("RungeKutta4")
    calculator.calculate(logEvery=5)
    print(calculator.xPrev)
    print(calculator.x)
    if i > 0:
        if calculator.xPrev[0] * calculator.x[0] <= 0.0:
            break

calculator.plot(axes,
                choice(NumericCalculation.PLOT_COLORS),
                label=r"$E={}$".format(e))

# extraArgs = (np.array([10**-5], dtype=np.float64),
#              np.array([1], dtype=np.float64), 20 / 5000, -20, 0, axes)
# print(fsolve(schrodinger_equation, -0.1, extraArgs, xtol=0.0000001))

axes.legend()
plt.show()
