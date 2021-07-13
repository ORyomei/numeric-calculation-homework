# from .numeric_calculation import NumericCalculation
from numeric_calculation import NumericCalculation
import numpy as np
import datetime
from os import makedirs
import matplotlib.pyplot as plt


def a(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.cross(v, x) / np.power(np.linalg.norm(x, ord=2), 3)


folderName = "data_{}".format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))
makedirs(folderName)

figure = plt.figure(figsize=(16, 7))
axes1 = figure.add_subplot(
    121,
    title="Orbit of a charged particle around a magnetic monopole",
    projection="3d",
    xlabel="$x$",
    ylabel="$y$",
    zlabel="$z$")
axes1.set_xlim([-0.1, 1.0])
axes1.set_ylim([-1.0, 1.0])
axes1.set_zlim([-0.5, 0.5])
axes1.view_init(elev=20, azim=70)
axes2 = figure.add_subplot(122,
                           title="$v_0$-dependence of minimum distance",
                           xlabel="$v_0$",
                           ylabel="Minimum distance")

colors = ["b", "g", "r", "c", "m", "y", "k"]
v_0s = [0.01, 0.1, 0.3, 0.5]
distanceMinimums = []

for v_0, color in zip(v_0s, colors):
    print("calculating v_0 = {} ...".format(v_0))
    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=np.array([1, 0, 0], dtype=np.float64),
                               vInitial=np.array([-1, v_0, 0],
                                                 dtype=np.float64),
                               tDelta=3.0 / 50000,
                               tInitial=0.0,
                               tFinal=3.0)
    calculator.setDataFileName("{}/data_{}.csv".format(
        folderName,
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f')))
    calculator.setMethod("RungeKutta4")
    calculator.setEquation(a)
    calculator.start(logEvery=500)
    calculator.plot(axes1,
                    color,
                    columns=[1, 2, 3],
                    label="$v_0 = {}$".format(v_0))
    distanceMinimums.append(calculator.distanceMinimum)

axes2.scatter(v_0s,
              distanceMinimums,
              marker=".",
              color="b",
              label="Minimum distance")
v = np.linspace(0.0, 0.6, 100)
d = v / np.power(1 + v * v, 0.5)
axes2.plot(v,
           d,
           marker=".",
           markersize=1.0,
           linewidth=1.0,
           color="r",
           label=r"$\frac{v_{0}}{\sqrt{1+v_{0}^{2}}}$")

axes2.legend()
axes1.legend()
plt.show()
