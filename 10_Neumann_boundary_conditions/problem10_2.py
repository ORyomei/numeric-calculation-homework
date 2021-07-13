from pde import BounderyCondition, PDE
import numpy as np
import matplotlib.pyplot as plt

J = 100
L = np.float64(1 / 3)
sigma = np.float64(1 / 20)
xDelta = np.float64(1 / J)
tDelta = np.power(xDelta, 2) * L / (2 * sigma)
t0 = np.float64(1 / 50)


def uFixedTInitialFunc(x: np.float64) -> np.float64:
    return np.exp(-np.power(x - 1, 2) / (4 * sigma * t0))


def derivativeOfT(
    t: np.float64,
    x: np.float64,
    u: np.float64,
    derivativeOfX: np.float64,
    secondDerivativeOfX: np.float64,
) -> np.float64:
    return sigma * secondDerivativeOfX


pde = PDE()
pde.setDataFileName("data10_1.csv")
pde.setLimits(
    xMin=np.float64(0),
    xMax=np.float64(1),
    xDelta=xDelta,
    tMin=np.float64(0),
    tMax=np.float64(5),
    tDelta=tDelta,
)
pde.setUFixedTInitial(uFixedTInitialFunc=uFixedTInitialFunc)
pde.setEquation(derivativeOfT)
pde.setBounderyCondition(BounderyCondition.NEUMANM, np.float64(0),
                         np.float64(0))
pde.calculate(tToLogs=[np.float64(t) for t in range(6)])

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$x$", ylabel="$u$")

pde.plot(axes, "b")

axes.legend()
plt.show()
