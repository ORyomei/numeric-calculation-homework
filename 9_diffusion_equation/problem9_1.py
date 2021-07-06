from pde import PDE
import numpy as np
import matplotlib.pyplot as plt

L = np.float64(1 / 3)
sigma = np.float64(0.1)
t0 = np.float64(1)


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


pde = PDE()
pde.setDataFileName("data9_1.csv")
pde.setLimits(
    xMin=np.float64(0),
    xMax=np.float64(10),
    xDelta=np.float64(0.1),
    tMin=np.float64(0),
    tMax=np.float64(5),
    tDelta=np.power(np.float64(0.1), 2) * L / (2 * sigma),
)
pde.setUFixedTInitial(uFixedTInitialFunc=uFixedTInitialFunc)
pde.setEquation(derivativeOfT)
pde.calculate(tToLogs=[np.float64(t) for t in range(6)])

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$x$", ylabel="$u$")

pde.plot(axes, "b", collums=[1, 2], label="$L={}$".format(round(L, 4)))

axes.legend()
plt.show()
