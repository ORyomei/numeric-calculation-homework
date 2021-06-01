# from .numeric_calculation import NumericCalculation
from numeric_calculation import NumericCalculation
import datetime
import numpy as np
import matplotlib.pyplot as plt


def a(x: np.ndarray, v: np.ndarray, epsilon: np.float64) -> np.ndarray:
    _a = 2.0 + epsilon * x[0] * x[0] - x[0]
    return np.array([_a], dtype=np.float64)


epsilons = [0.1 * np.power(0.7, i) for i in range(40)]
cycleTimes = []
for epsilon in epsilons:
    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=np.array([1], dtype=np.float64),
                               vInitial=np.array([0], dtype=np.float64),
                               tDelta=4 * np.pi / 50000,
                               tInitial=0.0,
                               tFinal=10 * np.pi)
    calculator.setMethod("Symplectic")

    def _a(x, v):
        return a(x, v, epsilon)

    calculator.setEquation(_a)
    calculator.start()
    cycleTime = (
        calculator.cycleTimes[-1] - calculator.cycleTimes[0]) / (len(calculator.cycleTimes) - 1)
    cycleTimes.append(cycleTime)

figure = plt.figure()
axes = figure.add_subplot(111)
# axes.set_xscale("log")
# axes.set_yscale("log")
axes.set_xlabel("$\epsilon$")
axes.set_ylabel("$T-2\pi$")
axes.plot(epsilons, [time - 2 * np.pi for time in cycleTimes], marker=".",
          markersize=2, linewidth=1.0, color="b")
# figure.savefig("plot4_2.png")

plt.show()
