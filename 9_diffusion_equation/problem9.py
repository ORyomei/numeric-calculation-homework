from typing import List
import numpy as np
import csv
import matplotlib.pyplot as plt

xMax = np.float64(10.0)
xMin = np.float64(0.0)
xDelta = np.float64(0.1)
L = np.float64(1 / 3)
sigma = np.float64(0.1)
tMax = np.float64(5.0)
tMin = np.float64(0.0)
tDelta = np.power(xDelta, 2) * L / (2 * sigma)
xIndexMax: int = int((xMax - xMin) / xDelta + 0.01)
tIndexMax: int = int((tMax - tMin) / tDelta + 0.01)
t0 = np.float64(1.0)
dataFileName = "data.csv"


def uFixedTInitial(xIndex: int) -> np.float64:
    x: np.float64 = xDelta * xIndex
    return np.exp(-np.power(x - 5, 2) /
                  (4 * sigma * t0)) / np.sqrt(4 * np.pi * sigma * t0)


def update(uFixedT: List[np.float64]):
    for xIndex in range(xIndexMax):
        uFixedT[xIndex] += sigma * (
            uFixedT[(xIndex + 1) %
                    (xIndexMax - 1)] - 2 * uFixedT[(xIndex) %
                                                   (xIndexMax - 1)] +
            uFixedT[(xIndex - 1) %
                    (xIndexMax - 1)]) * (tDelta / np.power(xDelta, 2))


uFixedT = [uFixedTInitial(i) for i in range(xIndexMax + 1)]
logEvery = 10
logTs = [np.float64(t) for t in range(6)]
with open(dataFileName, 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(uFixedT)
    for tIndex in range(1, tIndexMax + 1):
        update(uFixedT)
        t = tIndex * tDelta
        for logT in logTs:
            if np.abs(t - logT) < tDelta / 2:
                writer.writerow(uFixedT)

X = [xIndex * xDelta for xIndex in range(xIndexMax + 1)]

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$x$", ylabel="$u$")

with open(dataFileName, "r") as f:
    reader = csv.reader(f)
    for line in reader:
        u = [np.float64(value) for value in line]
        axes.plot(X, u, marker=".", markersize=0.0, linewidth=1.0, color="b")

plt.show()