from typing import List
import numpy as np
import csv
import matplotlib.pyplot as plt

xMax = np.float64(10.0)
xMin = np.float64(0.0)
xDelta = np.float64(0.02)
tMax = np.float64(3.0)
tMin = np.float64(0.0)
tDelta = np.float64(0.01)
xIndexMax: int = int((xMax - xMin) / xDelta + 0.01)
tIndexMax: int = int((tMax - tMin) / tDelta + 0.01)
dataFileName = "data.csv"

c = np.float64(-1)


def uFixedTInitial(xIndex: int) -> np.float64:
    x: np.float64 = xDelta * xIndex
    if 4 < x and x < 6:
        return (x - 4) * (6 - x)
    else:
        return np.float64(0.0)


def update(uFixedT: List[np.float64]):
    for xIndex in range(xIndexMax):
        uFixedT[xIndex] += -c * (uFixedT[xIndex + 1] -
                                 uFixedT[xIndex]) * (tDelta / xDelta)
    uFixedT[xIndexMax] = uFixedT[0]


uFixedT = [uFixedTInitial(i) for i in range(xIndexMax + 1)]
logEvery = 10

with open(dataFileName, 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(uFixedT)
    for tIndex in range(1, tIndexMax + 1):
        update(uFixedT)
        if tIndex % logEvery == 0:
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