import cmath
from typing import List
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def areaIllustrate(func, axes: plt.Axes, color: str, label: str):
    boundaryPolar = []
    for i in range(360):
        theta = 2 * np.pi * i / 360
        boundaryPolar.append(
            (optimize.fsolve(func, 5, args=(theta,))[0], theta,))
    boundaryX = [(cmath.rect(coor[0], coor[1]) -
                  0.5).real for coor in boundaryPolar]
    boundaryY = [(cmath.rect(coor[0], coor[1]) -
                  0.5).imag for coor in boundaryPolar]

    axes.set_aspect('equal', adjustable='box')
    axes.plot(boundaryX, boundaryY, marker='.',
              color=color, label=label)


def s(func, theta):
    def e(r: float):
        return func(r, theta)
    return e


def stableAreaRungeKutta2(r: float, theta: float) -> float:
    z = cmath.rect(r, theta) - 0.5
    Z = 1 + z + z ** 2 / 2
    return (Z * Z.conjugate() - 1).real


def stableAreaRungeKutta4(r: float, theta: float) -> float:
    z = cmath.rect(r, theta) - 0.5
    Z = 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
    return (Z * Z.conjugate() - 1).real


figure = plt.figure()
axes = figure.add_subplot(111)

areaIllustrate(stableAreaRungeKutta2, axes, "b", "RK2")
areaIllustrate(stableAreaRungeKutta4, axes, "r", "RK4")

axes.legend()
plt.show()
# figure.savefig("problem3.png")
