from typing import List, Literal
import numpy as np
import matplotlib.pyplot as plt
import csv


class NumericCalculation:
    fileName: str
    initialxAndv: np.ndarray
    xAndv: np.ndarray
    tDelta: np.float64
    t: np.float64
    tFinal: float
    method: str
    fileName: str
    cycleTimes: List[np.float64]
    distanceMinimum: np.float64

    def __init__(self):
        self.fileName = None

    def setInitialValue(self, xInitial: np.ndarray, vInitial: np.ndarray,
                        tDelta: np.float64, tInitial: float, tFinal: float):
        self.initialxAndv = np.vstack([xInitial, vInitial])
        self.xAndv = np.vstack([xInitial, vInitial])
        self.tDelta = tDelta
        self.t = tInitial
        self.tFinal = tFinal
        self.distanceMinimum = np.linalg.norm(self.x, ord=2)

    def setMethod(self, method: Literal["Euler", "RungeKutta2", "RungeKutta4",
                                        "Symplectic"]):
        if method == "Euler":
            self._update = self._updateEuler
        elif method == "RungeKutta2":
            self._update = self._updateRungeKutta2
        elif method == "RungeKutta4":
            self._update = self._updateRungeKutta4
        elif method == "Symplectic":
            self._update = self._updateSymplectic
        self.method = method

    def setEquation(self, func):
        self.a = func

    def setDataFileName(self, fileName: str):
        self.fileName = fileName

    def _dxAnddv(self, xAndv: np.ndarray) -> np.ndarray:
        return np.vstack([xAndv[1], self.a(xAndv[0], xAndv[1])])

    def _updateEuler(self):
        k1 = self._dxAnddv(self.xAndv)
        self.xAndv += self.tDelta * k1
        self.t += self.tDelta

    def _updateRungeKutta2(self):
        k1 = self._dxAnddv(self.xAndv)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1)
        self.xAndv += self.tDelta * (k1 + k2) / 2
        self.t += self.tDelta

    def _updateRungeKutta4(self):
        k1 = self._dxAnddv(self.xAndv)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1 / 2)
        k3 = self._dxAnddv(self.xAndv + self.tDelta * k2 / 2)
        k4 = self._dxAnddv(self.xAndv + self.tDelta * k3)
        self.xAndv += self.tDelta * (k1 + k2 * 2 + k3 * 2 + k4) / 6
        self.t += self.tDelta

    def _updateSymplectic(self):
        self.xAndv[0] += self.tDelta * self.xAndv[1]
        self.xAndv[1] += self.tDelta * self.a(self.xAndv[0], self.xAndv[1])
        self.t += self.tDelta

    def logData(self, writer):
        writer.writerow([self.t] + self.x.tolist())

    def plot(self,
             axes: plt.Axes,
             color: str,
             columns: List[int] = [0, 1],
             label=""):
        data = np.genfromtxt(self.fileName, delimiter=",")
        axes.plot(*[data[:, column] for column in columns],
                  marker=".",
                  markersize=2,
                  linewidth=1.0,
                  color=color,
                  label=label)

    def start(self, logEvery=1):
        if self.fileName:
            with open(self.fileName, 'a') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
                self.logData(writer)
                count = 0
                while True:
                    self._update()
                    distance = np.linalg.norm(self.x, ord=2)
                    if distance < self.distanceMinimum:
                        self.distanceMinimum = distance
                    if count % logEvery == 0:
                        self.logData(writer)
                    if self.t >= self.tFinal:
                        break
                    count += 1
        else:
            self.cycleTimes = []
            while True:
                self._update()
                if self.t >= self.tFinal:
                    break

    @property
    def x(self) -> np.ndarray:
        return self.xAndv[0]

    @property
    def v(self) -> np.ndarray:
        return self.xAndv[1]
