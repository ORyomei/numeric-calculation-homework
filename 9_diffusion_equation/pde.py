from typing import List
import numpy as np
import csv
import matplotlib.pyplot as plt


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

    def __init__(self):
        self.fileName = None

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
                          uFixedTInitialFunc=None):
        if uFixedTInitial != None:
            self.uFixedT = uFixedTInitial
            return
        if uFixedTInitialFunc != None:
            self.uFixedT = np.array([
                uFixedTInitialFunc(self.xMin + self.xDelta * xIndex)
                for xIndex in range(self.xIndexMax + 1)
            ])
            return

    def setEquation(self, func):
        self.derivativeOfT = func

    def _derivativeOfX(self, xIndex: int) -> np.float64:
        return (self.uFixedT[(xIndex + 1) % (self.xIndexMax)] -
                self.uFixedT[xIndex % (self.xIndexMax)]) / self.xDelta

    def _secondDerivativeOfX(self, xIndex: int) -> np.float64:
        return (self._derivativeOfX(xIndex) -
                self._derivativeOfX(xIndex - 1)) / self.xDelta

    def update(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            uDeltaFixedT[xIndex] = self.tDelta * (self.derivativeOfT(
                self.t, self.x(xIndex), u, self._derivativeOfX(xIndex),
                self._secondDerivativeOfX(xIndex)))
        self.uFixedT += uDeltaFixedT
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

    def plot(self,
             axes: plt.Axes,
             color: str,
             columns: List[int] = [0, 1],
             label=""):
        if self.fileName:
            data = np.genfromtxt(self.fileName, delimiter=",")
        else:
            data = np.array(self.coordinates)
        axes.plot(*[data[:, column] for column in columns],
                  marker=".",
                  markersize=0,
                  linewidth=1.0,
                  color=color,
                  label=label)

    def x(self, xIndex: int) -> np.float64:
        return self.xMin + self.xDelta * xIndex

    @property
    def t(self):
        return self.tDelta * self.tIndex