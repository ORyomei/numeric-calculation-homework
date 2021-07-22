from pde import BounderyCondition, PDE
import numpy as np
import matplotlib.pyplot as plt

L = np.float64(20)
sigma = np.float64(0.1)
t0 = np.float64(1)
tToLogs = [np.float64(t) for t in range(6)]


def uFixedTInitialFunc(x: np.float64) -> np.float64:
    return np.exp(-np.power(x - 5, 2) /
                  (4 * sigma * t0)) / np.sqrt(4 * np.pi * sigma * t0)


def analyticalSolution(x: np.float64, t: np.float64) -> np.float64:
    return np.exp(-np.power(x - 5, 2) /
                  (4 * sigma * (t + t0))) / np.sqrt(4 * np.pi * sigma *
                                                    (t + t0))


def dUdT(
    t: np.float64,
    x: np.float64,
    u: np.float64,
    dUdX: np.float64,
    d2UdX2: np.float64,
) -> np.float64:
    return sigma * d2UdX2


pde = PDE()
pde.setDataFileName("data11_1.csv")
pde.setLimits(
    xMin=np.float64(0),
    xMax=np.float64(10),
    xDelta=np.float64(0.1),
    tMin=np.float64(0),
    tMax=np.float64(5),
    tDelta=np.power(np.float64(0.1), 2) * L / (2 * sigma),
)
pde.setUFixedTInitial(uFixedTInitialFunc=uFixedTInitialFunc)
pde.setEquation(dUdT)
pde.setImplicity(np.float64(1))
pde.setBounderyCondition(BounderyCondition.DIRICHLET, np.float64(0),
                         np.float64(0))
pde.calculate(tToLogs=tToLogs)

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$x$", ylabel="$u$")

pde.plot(axes, "b", withLine=False)

xs = np.linspace(0, 10, 101)
for t in tToLogs:
    us = analyticalSolution(xs, t)
    axes.plot(xs,
              us,
              marker=".",
              markersize=0,
              linewidth=1.0,
              color="g",
              label=str(t))

plt.show()
