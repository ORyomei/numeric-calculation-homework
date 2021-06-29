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


def schrodingerEquation(eigenValue, xInitial: np.ndarray, vInitial: np.ndarray,
                        tDelta: np.float64, tInitial: float, tFinal: float,
                        axes: plt.Axes) -> np.float64:

    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=xInitial,
                               vInitial=vInitial,
                               tDelta=tDelta,
                               tInitial=tInitial,
                               tFinal=tFinal)
    distanceInitial = calculator.distance

    def _a(x: np.ndarray, v: np.ndarray, t: np.float64) -> np.ndarray:
        return a(x, t, eigenValue)

    calculator.setEquation(_a)
    calculator.setMethod("RungeKutta4")
    calculator.calculate(logEvery=1)
    # if calculator.distance <= distanceInitial * 10000000:
    calculator.plot(axes,
                    choice(NumericCalculation.PLOT_COLORS),
                    label=r"$E={}$".format(eigenValue))
    return np.linalg.norm(calculator.x, ord=2)


figure = plt.figure(figsize=(16, 7))
axes = figure.add_subplot(111)

extraArgs = (np.array([10**-5], dtype=np.float64),
             np.array([1], dtype=np.float64), 20 / 5000, -20, 0, axes)
print(fsolve(schrodingerEquation, -0.1, extraArgs, xtol=0.0000001))

axes.legend()
plt.show()
