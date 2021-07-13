from typing import Callable, List
import numpy as np
import csv
import matplotlib.pyplot as plt
from enum import IntEnum


class BounderyCondition(IntEnum):
    CYCLIC = 0
    DIRICHLET = 1
    NEUMANM = 2


class PDE:
    fileName: str
    xMin: np.float64
    xMax: np.float64
    xDelta: np.float64
    tMin: np.float64
    tMax: np.float64
    tDelta: np.float64
    xIndexMax: int
    tIndexMax: int
    tIndex: int
    uFixedT: np.ndarray
    bounderyCondition: BounderyCondition
    update: Callable[[], None]
    derivativeOfT: Callable[
        [np.float64, np.float64, np.float64, np.float64, np.float64],
        np.float64]
    uXMin: np.float64
    uXMax: np.float64
    derivativeOfXXMin: np.float64
    derivativeOfXXMax: np.float64
    tLogged: List[np.float64]

    def __init__(self):
        self.fileName = None
        self.bounderyCondition = BounderyCondition.CYCLIC
        self.update = self._updateCyclic
        self.tLogged = []

    def setLimits(self, xMin: np.float64, xMax: np.float64, xDelta: np.float64,
                  tMin: np.float64, tMax: np.float64, tDelta: np.float64):
        self.xMin = xMin
        self.xMax = xMax
        self.xDelta = xDelta
        self.tMin = tMin
        self.tMax = tMax
        self.tIndex = 0
        self.tDelta = tDelta
        self.xIndexMax = int((xMax - xMin) / xDelta + 0.001)
        self.tIndexMax = int((tMax - tMin) / tDelta + 0.001)

    def setUFixedTInitial(self,
                          uFixedTInitial: np.ndarray = None,
                          uFixedTInitialFunc: Callable[[np.float64],
                                                       np.float64] = None):
        if uFixedTInitial != None:
            self.uFixedT = uFixedTInitial
            return
        if uFixedTInitialFunc != None:
            self.uFixedT = np.array([
                uFixedTInitialFunc(self.x(xIndex))
                for xIndex in range(self.xIndexMax + 1)
            ])
            return

    def setEquation(self, func: Callable[
        [np.float64, np.float64, np.float64, np.float64, np.float64],
        np.float64]):
        self.derivativeOfT = func

    def setBounderyCondition(self, bounderyCondition: BounderyCondition,
                             value1: np.float64, value2: np.float64):
        self.bounderyCondition = bounderyCondition
        if bounderyCondition == BounderyCondition.CYCLIC:
            self.update = self._updateCyclic
        elif bounderyCondition == BounderyCondition.DIRICHLET:
            self.uXMin = value1
            self.uXMax = value2
            self.update = self._updateDirichlet
        elif bounderyCondition == BounderyCondition.NEUMANM:
            self.derivativeOfXXMin = value1
            self.derivativeOfXXMax = value2
            self.update = self._updateNeumann

    def _derivativeOfXCyclic(self, xIndex: int) -> np.float64:
        return (self.uFixedT[(xIndex + 1) % (self.xIndexMax)] -
                self.uFixedT[xIndex % (self.xIndexMax)]) / self.xDelta

    def _derivativeOfX(self, xIndex: int) -> np.float64:
        return (self.uFixedT[xIndex + 1] - self.uFixedT[xIndex]) / self.xDelta

    def _secondDerivativeOfXCyclic(self, xIndex: int) -> np.float64:
        return (self._derivativeOfXCyclic(xIndex) -
                self._derivativeOfXCyclic(xIndex - 1)) / self.xDelta

    def _secondDerivativeOfX(self, xIndex: int) -> np.float64:
        return (self._derivativeOfX(xIndex) -
                self._derivativeOfX(xIndex - 1)) / self.xDelta

    def _updateCyclic(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            uDeltaFixedT[xIndex] = self.tDelta * (self.derivativeOfT(
                self.t, self.x(xIndex), u, self._derivativeOfXCyclic(xIndex),
                self._secondDerivativeOfXCyclic(xIndex)))
        self.uFixedT += uDeltaFixedT
        self.tIndex += 1

    def _updateDirichlet(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            if xIndex == 0 or xIndex == self.xIndexMax:
                uDeltaFixedT[xIndex] = np.float64(0)
            else:
                uDeltaFixedT[xIndex] = self.tDelta * self.derivativeOfT(
                    self.t, self.x(xIndex), u, self._derivativeOfX(xIndex),
                    self._secondDerivativeOfX(xIndex))
        self.uFixedT += uDeltaFixedT
        self.uFixedT[0] = self.uXMin
        self.uFixedT[self.xIndexMax] = self.uXMax
        self.tIndex += 1

    def _updateNeumann(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            if xIndex == 0 or xIndex == self.xIndexMax:
                uDeltaFixedT[xIndex] = np.float64(0)
            else:
                uDeltaFixedT[xIndex] = self.tDelta * self.derivativeOfT(
                    self.t, self.x(xIndex), u, self._derivativeOfX(xIndex),
                    self._secondDerivativeOfX(xIndex))
        self.uFixedT += uDeltaFixedT
        self.uFixedT[0] = (self.uFixedT[1] * 4 - self.uFixedT[2]
                           ) / 3 - 2 * self.xDelta * self.derivativeOfXXMin / 3
        self.uFixedT[self.xIndexMax] = (
            self.uFixedT[self.xIndexMax - 1] * 4 -
            self.uFixedT[self.xIndexMax -
                         2]) / 3 + 2 * self.xDelta * self.derivativeOfXXMax / 3
        self.tIndex += 1

    def calculate(self, logEvery: int = 1, tToLogs: List[np.float64] = None):
        with open(self.fileName, "w") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            if tToLogs != None:
                self._calculateWithTToLogs(tToLogs, writer)
                return
            else:
                self._calculateWithLogEvery(logEvery, writer)

    def _calculateWithLogEvery(self, logEvery: int, writer):
        self._logData(writer)
        for tIndex in range(self.tIndexMax):
            self.update()
            if tIndex % logEvery == 0:
                self._logData(writer)

    def _calculateWithTToLogs(self, tToLogs: List[np.float64], writer):
        if self._isToLog(tToLogs):
            self._logData(writer)
        for tIndex in range(self.tIndexMax):
            self.update()
            if self._isToLog(tToLogs):
                self._logData(writer)

    def _isToLog(self, tToLogs: List[np.float64]) -> bool:
        for tToLog in tToLogs:
            if np.abs(self.t - tToLog) < self.tDelta / 2:
                return True
        return False

    def setDataFileName(self, fileName: str):
        self.fileName = fileName

    def _logData(self, writer):
        for xIndex, u in enumerate(list(self.uFixedT)):
            writer.writerow([self.t, self.x(xIndex), u])
        self.tLogged.append(self.t)

    def plot(self, axes: plt.Axes, color: str):
        data = np.genfromtxt(self.fileName, delimiter=",")
        uTX = [[[], []] for t in self.tLogged]
        for point in data:
            uTX[self.tLogged.index(point[0])][0].append(point[1])
            uTX[self.tLogged.index(point[0])][1].append(point[2])

        for i, t in enumerate(self.tLogged):
            axes.plot(uTX[i][0],
                      uTX[i][1],
                      marker=".",
                      markersize=0,
                      linewidth=1.0,
                      color=color,
                      label=str(t))

    def x(self, xIndex: int) -> np.float64:
        return self.xMin + self.xDelta * xIndex

    @property
    def t(self):
        return self.tDelta * self.tIndex