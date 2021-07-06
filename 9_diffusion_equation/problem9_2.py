from typing import List
from pde import PDE
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs

sigma = np.float64(0.1)
t0 = np.float64(1)
te = np.float64(6)
xDelta = np.float64(0.1)

Ls = [np.float64(L) for L in [4 / 5, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10]]


def f(x: np.float64) -> np.float64:
    return np.exp(-np.power(x - 5, 2) /
                  (4 * sigma * te)) / np.sqrt(4 * np.pi * sigma * te)


def uFixedTInitialFunc(x: np.float64) -> np.float64:
    return np.exp(-np.power(x - 5, 2) /
                  (4 * sigma * t0)) / np.sqrt(4 * np.pi * sigma * t0)


def derivativeOfT(
    t: np.float64,
    x: np.float64,
    u: np.float64,
    derivativeOfX: np.float64,
    secondDerivativeOfX: np.float64,
) -> np.float64:
    return sigma * secondDerivativeOfX


folderName = "data9_2"
makedirs(folderName, exist_ok=True)


def rmse(uFixedT: List[np.float64]) -> np.float64:
    error = np.float64(0)
    for xIndex, u in enumerate(uFixedT):
        error += np.power((u - f(xIndex * xDelta)), 2)
    return xDelta * error


eErrors = []

for L in Ls:
    pde = PDE()
    dataFileName = "{}/data_L_{}.csv".format(
        folderName,
        str(round(L, 4)).replace(".", "_"))
    pde.setDataFileName(dataFileName)
    pde.setLimits(
        xMin=np.float64(0),
        xMax=np.float64(10),
        xDelta=xDelta,
        tMin=np.float64(0),
        tMax=np.float64(5),
        tDelta=np.power(np.float64(0.1), 2) * L / (2 * sigma),
    )
    pde.setUFixedTInitial(uFixedTInitialFunc=uFixedTInitialFunc)
    pde.setEquation(derivativeOfT)
    pde.calculate(tToLogs=[np.float64(5)])

    uFixedT = np.genfromtxt(dataFileName, delimiter=",")[:, 2]
    eErrors.append(rmse(uFixedT))

figure = plt.figure(figsize=(9, 7))
axes = figure.add_subplot(111, xlabel="$L$", ylabel="$E_{error}$")
axes.set_yscale("log")
axes.set_xscale("log")
axes.plot(Ls, eErrors, marker=".", markersize=0, linewidth=1.0, color="b")

plt.show()