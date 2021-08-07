from os import write
from typing import Callable, List, Literal
import numpy as np
import matplotlib.pyplot as plt
import csv


class NumericCalculation:
    fileName: str
    initialxAndv: np.ndarray
    xAndv: np.ndarray
    xAndvPrev: np.ndarray
    tDelta: np.float64
    t: np.float64
    tFinal: float
    method: str
    fileName: str
    cycleTimes: List[np.float64]
    distanceMinimum: np.float64
    distanceMaximum: np.float64
    coordinates: np.ndarray
    dimension: int
    xHigherLimit: np.ndarray
    xLowerLimit: np.ndarray
    a: Callable[[np.ndarray, np.ndarray, np.float64], np.ndarray]
    _updateNumeric: Callable[[], None]

    PLOT_COLORS = ["b", "g", "r", "c", "m", "y", "k"]

    def __init__(self):
        self.fileName = None

    def setInitialValue(self, xInitial: np.ndarray, vInitial: np.ndarray,
                        tDelta: np.float64, tInitial: np.float64,
                        tFinal: np.float64):
        self.initialxAndv = np.vstack([xInitial, vInitial])
        self.xAndv = np.vstack([xInitial, vInitial])
        self.tDelta = tDelta
        self.t = tInitial
        self.tFinal = tFinal
        self.distanceMinimum = np.linalg.norm(self.x, ord=2)
        self.distanceMaximum = np.linalg.norm(self.x, ord=2)
        self.dimension = len(xInitial)
        self.xHigherLimit = np.full(self.dimension, np.finfo(np.float64).max)
        self.xLowerLimit = np.full(self.dimension, np.finfo(np.float64).min)

    def setMethod(self, method: Literal["Euler", "RungeKutta2", "RungeKutta4",
                                        "Symplectic"]):
        if method == "Euler":
            self._updateNumeric = self._updateEuler
        elif method == "RungeKutta2":
            self._updateNumeric = self._updateRungeKutta2
        elif method == "RungeKutta4":
            self._updateNumeric = self._updateRungeKutta4
        elif method == "Symplectic":
            self._updateNumeric = self._updateSymplectic
        self.method = method

    def setEquation(self, func: Callable[[np.ndarray, np.ndarray, np.float64],
                                         np.ndarray]):
        self.a = func

    def setDataFileName(self, fileName: str):
        self.fileName = fileName

    def setLimit(self, xLowerLimit: np.ndarray,
                 xHigherLimit: np.ndarray) -> bool:
        for (lowerLimit, higherLimit) in zip(xLowerLimit, xHigherLimit):
            if lowerLimit >= higherLimit:
                print("invalid Limit")
                return False
        self.xLowerLimit = xLowerLimit
        self.xHigherLimit = xHigherLimit
        return True

    def _isInLimit(self) -> bool:
        for (value, lowerLimit, higherLimit) in zip(self.x, self.xLowerLimit,
                                                    self.xHigherLimit):
            if value < lowerLimit or higherLimit < value:
                return False
        return True

    def _isToFinish(self) -> bool:
        tIsFinal = np.abs(self.t - self.tFinal) <= self.tDelta / 2
        outOfLimit = not self._isInLimit
        return tIsFinal or outOfLimit

    def _dxAnddv(self, xAndv: np.ndarray, t: np.float64) -> np.ndarray:
        return np.vstack([xAndv[1], self.a(xAndv[0], xAndv[1], t)])

    def _updateEuler(self):
        self.xAndvPrev = np.copy(self.xAndv)
        k1 = self._dxAnddv(self.xAndv, self.t)
        self.xAndv += self.tDelta * k1
        self.t += self.tDelta

    def _updateRungeKutta2(self):
        self.xAndvPrev = np.copy(self.xAndv)
        k1 = self._dxAnddv(self.xAndv, self.t)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1, self.t + self.tDelta)
        self.xAndv += self.tDelta * (k1 + k2) / 2
        self.t += self.tDelta

    def _updateRungeKutta4(self):
        self.xAndvPrev = np.copy(self.xAndv)
        k1 = self._dxAnddv(self.xAndv, self.t)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1 / 2,
                           self.t + self.tDelta / 2)
        k3 = self._dxAnddv(self.xAndv + self.tDelta * k2 / 2,
                           self.t + self.tDelta / 2)
        k4 = self._dxAnddv(self.xAndv + self.tDelta * k3, self.t + self.tDelta)
        self.xAndv += self.tDelta * (k1 + k2 * 2 + k3 * 2 + k4) / 6
        self.t += self.tDelta

    def _updateSymplectic(self):
        self.xAndvPrev = np.copy(self.xAndv)
        self.xAndv[0] += self.tDelta * self.xAndv[1]
        self.xAndv[1] += self.tDelta * self.a(self.xAndv[0], self.xAndv[1],
                                              self.t)
        self.t += self.tDelta

    def _updateDistance(self):
        if self.distance < self.distanceMinimum:
            self.distanceMinimum = self.distance
        if self.distance > self.distanceMaximum:
            self.distanceMaximum = self.distance

    def update(self):
        self._updateNumeric()
        self._updateDistance()

    def _logData(self, writer=None):
        if writer != None:
            writer.writerow([self.t] + self.x.tolist())
        else:
            self.coordinates = np.block(
                [[self.coordinates], [np.array([self.t] + self.x.tolist())]])

    def plot(self,
             axes: plt.Axes,
             color: str,
             columns: List[int] = [0, 1],
             label=""):
        if self.fileName:
            data = np.genfromtxt(self.fileName, delimiter=",")
        else:
            data = self.coordinates
        axes.plot(*[data[:, column] for column in columns],
                  marker=".",
                  markersize=0,
                  linewidth=1.0,
                  color=color,
                  label=label)

    def calculate(self, logEvery: int = 1, logEveryT: np.float64 = None):
        if self.fileName:
            self._calculateToOutputFile(logEvery, logEveryT)
        else:
            self._calculateToList(logEvery, logEveryT)

    def _isToLog(self,
                 count: int,
                 logEvery: int = 1,
                 logEveryT: np.float64 = None) -> bool:
        if logEveryT == None:
            return count % logEvery == 0
        else:
            return np.abs((self.t + logEveryT / 2) % logEveryT -
                          logEveryT / 2) < self.tDelta / 2

    def _calculateToOutputFile(self, logEvery: int, logEveryT: np.float64):
        with open(self.fileName, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            count = 0
            if self._isToLog(count, logEvery, logEveryT):
                self._logData(writer)
            while True:
                self.update()
                count += 1
                if self._isToLog(count, logEvery, logEveryT):
                    self._logData(writer)
                if self._isToFinish():
                    break

    def _calculateToList(self, logEvery: int, logEveryT: np.float64):
        self.coordinates = np.empty((0, 2), dtype=np.float64)
        count = 0
        if self._isToLog(count, logEvery, logEveryT):
            self._logData()
        while True:
            self.update()
            count += 1
            if self._isToLog(count, logEvery, logEveryT):
                self._logData()
            if self._isToFinish():
                break

    def normalize(self):
        self.coordinates[:, 1:self.dimension + 1] /= self.distanceMaximum

    @property
    def x(self) -> np.ndarray:
        return self.xAndv[0]

    @property
    def v(self) -> np.ndarray:
        return self.xAndv[1]

    @property
    def xPrev(self) -> np.ndarray:
        return self.xAndvPrev[0]

    @property
    def vPrev(self) -> np.ndarray:
        return self.xAndvPrev[1]

    @property
    def distance(self) -> np.float64:
        return np.linalg.norm(self.x, ord=2)
