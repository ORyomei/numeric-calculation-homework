from typing import List
import numpy as np
import csv
import matplotlib.pyplot as plt

dx = np.float64(1) / 20
dt = np.float64(np.pi) / 3200
tMax = np.float64(2) * np.pi
xMax = np.float64(10)
xMin = np.float64(-10)
jMax = int((xMax - xMin) / dx + 0.00001)
nMax = int((tMax - 0) / dt + 0.00001)
tToLogs = [np.float64(np.pi) * k / 4 for k in range(9)]

x0 = np.float64(5)
dataFileName = "data12_1.csv"


def t(n: int) -> np.float64:
    return n * dt


def x(j: int) -> np.float64:
    return xMin + j * dx


xs = [x(j) for j in range(jMax + 1)]


def V(x: np.float64) -> np.float64:
    return x * x / 2


def dPsidX(values: np.ndarray, j: int) -> np.float64:
    return (values[(j + 1) % jMax] - values[j % jMax]) / dx


def d2PsidX2(values: np.ndarray, j: int) -> np.float64:
    return (dPsidX(values, j) - dPsidX(values, j - 1)) / dx


def H(values: np.ndarray, j: int) -> np.float64:
    return -d2PsidX2(values, j) / 2 + V(x(j)) * values[j]


def initialReal(x: np.float64) -> np.float64:
    return np.exp(-(x - x0) * (x - x0) / 2) / np.power(np.pi, 1 / 4)


def updateReal(real, imag):
    return real + dt * np.array([H(imag, j) for j in range(jMax + 1)])


def updateImag(real, imag):
    return imag + dt * np.array([-H(real, j) for j in range(jMax + 1)])


def isToLog(n: int) -> bool:
    for tToLog in tToLogs:
        if np.abs(t(n) - tToLog) < dt / 2:
            return True
    return False


def logData(real: np.ndarray, imag: np.ndarray, imagPrev: np.ndarray):
    with open(dataFileName, "a") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(probabilityDensity(real, imag, imagPrev))


def probabilityDensity(real: np.ndarray, imag: np.ndarray,
                       imagPrev: np.ndarray) -> np.ndarray:
    return real * real + imag * imagPrev


real = np.array([initialReal(x(j)) for j in range(jMax + 1)])
imag = np.zeros(jMax + 1, dtype=np.float64)
n = 0

if isToLog(n):
    logData(real, imag, imag)

for n in range(nMax):
    imagPrev = np.copy(imag)
    real = updateReal(real, imag)
    imag = updateImag(real, imag)
    n += 1
    if isToLog(n):
        logData(real, imag, imagPrev)

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$x$", ylabel="$\psi$")

colors = ["b", "g", "r", "c", "m", "y", "k", "r", "c"]
with open(dataFileName, "r") as f:
    reader = csv.reader(f)
    for t_, line, color in zip(tToLogs, reader, colors):
        psis = [np.float64(value) for value in line]
        axes.plot(xs,
                  psis,
                  marker=".",
                  markersize=0.0,
                  linewidth=1.0,
                  color=color,
                  label="$t={}$".format(round(t_, 3)))

axes.legend()
plt.show()