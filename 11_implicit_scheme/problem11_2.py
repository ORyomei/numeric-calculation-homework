from pde import BounderyCondition, PDE
import numpy as np
import matplotlib.pyplot as plt

L = np.float64(5) / 3
sigma = np.float64(0.1)
t0 = np.float64(1)
implicities = [np.float64(0.2 + 0.01 * i) for i in range(31)]
xDelta = np.float64(0.1)
tMax = np.float64(5)


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


Es = []

for implicity in implicities:
    pde = PDE()
    pde.setLimits(
        xMin=np.float64(0),
        xMax=np.float64(10),
        xDelta=xDelta,
        tMin=np.float64(0),
        tMax=tMax,
        tDelta=np.power(np.float64(0.1), 2) * L / (2 * sigma),
    )
    pde.setUFixedTInitial(uFixedTInitialFunc=uFixedTInitialFunc)
    pde.setEquation(dUdT)
    pde.setImplicity(implicity)
    pde.setBounderyCondition(BounderyCondition.DIRICHLET, np.float64(0),
                             np.float64(0))
    pde.calculate(noLogging=True)
    E = xDelta * sum([
        np.power(u - analyticalSolution(i * xDelta, tMax), 2)
        for i, u in enumerate(pde.uFixedT)
    ])
    Es.append(E)

figure = plt.figure(figsize=(9, 7))
axes = figure.add_subplot(111, xlabel=r"$\theta$", ylabel="error")

axes.plot(implicities, Es, marker=".", markersize=5, linewidth=0, color="b")

plt.show()
