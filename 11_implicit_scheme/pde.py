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
    dUdT: Callable[
        [np.float64, np.float64, np.float64, np.float64, np.float64],
        np.float64]
    uXMin: np.float64
    uXMax: np.float64
    dUdXXMin: np.float64
    dUdXXMax: np.float64
    tLogged: List[np.float64]
    implicity: np.float64
    coefficientU: Callable[[np.float64, np.float64], np.float64]
    coefficientdUdT: Callable[[np.float64, np.float64], np.float64]
    coefficientd2UdT2: Callable[[np.float64, np.float64], np.float64]
    constant: Callable[[np.float64, np.float64], np.float64]

    def __init__(self):
        self.fileName = None
        self.bounderyCondition = BounderyCondition.CYCLIC
        self.update = self._updateCyclic
        self.tLogged = []
        self.implicity = np.float64(0)

    def setLimits(self, xMin: np.float64, xMax: np.float64, xDelta: np.float64,
                  tMin: np.float64, tMax: np.float64, tDelta: np.float64):
        self.xMin = xMin
        self.xMax = xMax
        self.xDelta = xDelta
        self.tMin = tMin
        self.tMax = tMax
        self.tIndex = 0
        self.tDelta = tDelta
        self.xIndexMax = int((xMax - xMin) / xDelta + 0.00001)
        self.tIndexMax = int((tMax - tMin) / tDelta + 0.00001)

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
        self.dUdT = func

    def setBounderyCondition(self,
                             bounderyCondition: BounderyCondition,
                             value1: np.float64 = np.float64(0),
                             value2: np.float64 = np.float64(0)):
        self.bounderyCondition = bounderyCondition
        if bounderyCondition == BounderyCondition.DIRICHLET:
            self.uXMin = value1
            self.uXMax = value2
        elif bounderyCondition == BounderyCondition.NEUMANM:
            self.dUdXXMin = value1
            self.dUdXXMax = value2
        self._setUpdator()

    def setImplicity(self, implicity: np.float64):
        """
        陰的スキームは微分方程式が線形の場合のみ使用できます。
        """
        self.implicity = implicity
        self._setUpdator()

    def _setUpdator(self):
        if self.bounderyCondition == BounderyCondition.CYCLIC:
            if self.implicity == np.float64(0):
                self.update = self._updateCyclic
            else:
                self.update = self._updateCyclicImplicit
        elif self.bounderyCondition == BounderyCondition.DIRICHLET:
            if self.implicity == np.float64(0):
                self.update = self._updateDirichlet
            else:
                self.update = self._updateDirichletImplicit
        elif self.bounderyCondition == BounderyCondition.NEUMANM:
            if self.implicity == np.float64(0):
                self.update = self._updateNeumann
            else:
                self.update = self._updateNeumannImplicit

    def setdUdTCoefficients(
            self, coefficientU: Callable[[np.float64, np.float64], np.float64],
            coefficientdUdT: Callable[[np.float64, np.float64], np.float64],
            coefficientd2UdT2: Callable[[np.float64, np.float64], np.float64],
            constant: Callable[[np.float64, np.float64], np.float64]):
        self.coefficientU = coefficientU
        self.coefficientdUdT = coefficientdUdT
        self.coefficientd2UdT2 = coefficientd2UdT2
        self.constant = constant

        def dUdT(t: np.float64, x: np.float64, u: np.float64, dUdX: np.float64,
                 d2UdX2: np.float64):
            coefficientU(t, x) * u + coefficientdUdT(
                t, x) * dUdX + coefficientd2UdT2(t, x) * d2UdX2 + constant(
                    t, x)

        self.dUdT = dUdT

    def _dUdXCyclic(self, xIndex: int) -> np.float64:
        return (self.uFixedT[(xIndex + 1) % (self.xIndexMax)] -
                self.uFixedT[xIndex % (self.xIndexMax)]) / self.xDelta

    def _dUdX(self, xIndex: int) -> np.float64:
        return (self.uFixedT[xIndex + 1] - self.uFixedT[xIndex]) / self.xDelta

    def _d2UdX2Cyclic(self, xIndex: int) -> np.float64:
        return (self._dUdXCyclic(xIndex) -
                self._dUdXCyclic(xIndex - 1)) / self.xDelta

    def _d2UdX2(self, xIndex: int) -> np.float64:
        return (self._dUdX(xIndex) - self._dUdX(xIndex - 1)) / self.xDelta

    def _updateCyclic(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            uDeltaFixedT[xIndex] = self.tDelta * (self.dUdT(
                self.t, self.x(xIndex), u, self._dUdXCyclic(xIndex),
                self._d2UdX2Cyclic(xIndex)))
        self.uFixedT += uDeltaFixedT
        self.tIndex += 1

    def _updateDirichlet(self):
        uDeltaFixedT = np.zeros(self.xIndexMax + 1, dtype=np.float64)
        for xIndex, u in enumerate(self.uFixedT):
            if xIndex == 0 or xIndex == self.xIndexMax:
                uDeltaFixedT[xIndex] = np.float64(0)
            else:
                uDeltaFixedT[xIndex] = self.tDelta * self.dUdT(
                    self.t, self.x(xIndex), u, self._dUdX(xIndex),
                    self._d2UdX2(xIndex))
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
                uDeltaFixedT[xIndex] = self.tDelta * self.dUdT(
                    self.t, self.x(xIndex), u, self._dUdX(xIndex),
                    self._d2UdX2(xIndex))
        self.uFixedT += uDeltaFixedT
        self.uFixedT[0] = (self.uFixedT[1] * 4 - self.uFixedT[2]
                           ) / 3 - 2 * self.xDelta * self.dUdXXMin / 3
        self.uFixedT[
            self.xIndexMax] = (self.uFixedT[self.xIndexMax - 1] * 4 -
                               self.uFixedT[self.xIndexMax - 2]
                               ) / 3 + 2 * self.xDelta * self.dUdXXMax / 3
        self.tIndex += 1

    def _p(self, xIndex: int) -> np.float64:
        O = np.float64(0)
        t_ = self.tNext
        dx2 = np.power(self.xDelta, 2)
        dt = self.tDelta
        x = self.x(xIndex)
        return -self.implicity * (self.dUdT(t_, x, O, O, dt / dx2) -
                                  self.dUdT(t_, x, O, O, O))

    def _q(self, xIndex: int) -> np.float64:
        O = np.float64(0)
        t_ = self.tNext
        dx = self.xDelta
        dx2 = np.power(self.xDelta, 2)
        dt = self.tDelta
        x = self.x(xIndex)
        return 1 + self.implicity * (self.dUdT(
            t_, x, -dt, dt / dx, 2 * dt / dx2) - self.dUdT(t_, x, O, O, O))

    def _r(self, xIndex: int) -> np.float64:
        O = np.float64(0)
        t_ = self.tNext
        dx = self.xDelta
        dx2 = np.power(self.xDelta, 2)
        dt = self.tDelta
        x = self.x(xIndex)
        return -self.implicity * (self.dUdT(t_, x, O, dt / dx, dt / dx2) -
                                  self.dUdT(t_, x, O, O, O))

    def _s(self, xIndex: int) -> np.float64:
        O = np.float64(0)
        t_ = self.tNext
        t = self.t
        dt = self.tDelta
        x = self.x(xIndex)
        return self.uFixedT[xIndex] + dt * self.implicity * self.dUdT(
            t_, x, O, O, O) + dt * (1 - self.implicity) * self.dUdT(
                t, x, self.uFixedT[xIndex], self._dUdX(xIndex),
                self._d2UdX2(xIndex))

    def _implicitCoefficientMatrixCyclic(self) -> np.matrix:
        coefficientMatrix: np.matrix
        coefficientMatrix = np.matrix(np.zeros(
            (self.xIndexMax, self.xIndexMax)),
                                      dtype=np.float64)
        for xIndex in range(self.xIndexMax):
            coefficientMatrix[xIndex,
                              (xIndex - 1) % self.xIndexMax] = self._p(xIndex)
            coefficientMatrix[xIndex, xIndex] = self._q(xIndex)
            coefficientMatrix[xIndex,
                              (xIndex + 1) % self.xIndexMax] = self._r(xIndex)
        return coefficientMatrix

    def _implicitCoefficientVectorCyclic(self) -> np.ndarray:
        return np.array([self._s(xIndex) for xIndex in range(self.xIndexMax)])

    def _implicitCoefficientMatrixDirichlet(self) -> np.matrix:
        coefficientMatrix: np.matrix
        coefficientMatrix = np.matrix(np.zeros(
            (self.xIndexMax - 1, self.xIndexMax - 1)),
                                      dtype=np.float64)
        for xIndex in range(1, self.xIndexMax):
            index = xIndex - 1
            if xIndex - 1 in range(1, self.xIndexMax):
                coefficientMatrix[index, index - 1] = self._p(xIndex)
            coefficientMatrix[index, index] = self._q(xIndex)
            if xIndex + 1 in range(1, self.xIndexMax):
                coefficientMatrix[index, index + 1] = self._r(xIndex)
        return coefficientMatrix

    def _implicitCoefficientVectorDirichlet(self) -> np.ndarray:
        coefficientVector = np.array(
            [self._s(xIndex) for xIndex in range(1, self.xIndexMax)])
        coefficientVector[0] += -self._p(1) * self.uXMin
        coefficientVector[self.xIndexMax -
                          2] += -self._p(self.xIndexMax - 1) * self.uXMax
        return coefficientVector

    def _implicitCoefficientMatrixNeumann(self) -> np.matrix:
        coefficientMatrix: np.matrix
        coefficientMatrix = np.matrix(np.zeros(
            (self.xIndexMax - 1, self.xIndexMax - 1)),
                                      dtype=np.float64)
        for xIndex in range(1, self.xIndexMax):
            index = xIndex - 1
            if xIndex - 1 in range(1, self.xIndexMax):
                coefficientMatrix[index, index - 1] = self._p(xIndex)
            coefficientMatrix[index, index] = self._q(xIndex)
            if xIndex + 1 in range(1, self.xIndexMax):
                coefficientMatrix[index, index + 1] = self._r(xIndex)
        coefficientMatrix[0, 0] += 4 * self._p(1) / 3
        coefficientMatrix[0, 1] += -self._p(1) / 3
        coefficientMatrix[self.xIndexMax - 2, self.xIndexMax -
                          3] += -self._p(self.xIndexMax - 1) / 3
        coefficientMatrix[self.xIndexMax - 2, self.xIndexMax -
                          2] += 4 * self._p(self.xIndexMax - 1) / 3
        return coefficientMatrix

    def _implicitCoefficientVectorNeumann(self) -> np.ndarray:
        coefficientVector = np.array(
            [self._s(xIndex) for xIndex in range(1, self.xIndexMax)])
        coefficientVector[0] += 2 * self._p(
            1) * self.xDelta * self.dUdXXMin / 3
        coefficientVector[self.xIndexMax - 2] += -2 * self._p(
            self.xIndexMax - 1) * self.xDelta * self.dUdXXMax / 3
        return coefficientVector

    def _updateCyclicImplicit(self):
        coefficientMatrix = self._implicitCoefficientMatrixCyclic()
        coefficientVector = self._implicitCoefficientVectorCyclic()
        _uFixedT = np.linalg.solve(coefficientMatrix, coefficientVector)
        self.uFixedT = np.append(_uFixedT, _uFixedT[0])
        self.tIndex += 1

    def _updateDirichletImplicit(self):
        coefficientMatrix = self._implicitCoefficientMatrixDirichlet()
        coefficientVector = self._implicitCoefficientVectorDirichlet()
        _uFixedT = np.linalg.solve(coefficientMatrix, coefficientVector)
        self.uFixedT = np.append(np.append(self.uXMin, _uFixedT), self.uXMax)
        self.tIndex += 1

    def _updateNeumannImplicit(self):
        coefficientMatrix = self._implicitCoefficientMatrixDirichlet()
        coefficientVector = self._implicitCoefficientVectorDirichlet()
        _uFixedT = np.linalg.solve(coefficientMatrix, coefficientVector)
        uXMin = (_uFixedT[0] * 4 -
                 _uFixedT[1]) / 3 - 2 * self.xDelta * self.dUdXXMin / 3
        uXMax = (_uFixedT[self.xIndexMax - 2] * 4 - _uFixedT[
            self.xIndexMax - 3]) / 3 + 2 * self.xDelta * self.dUdXXMax / 3
        self.uFixedT = np.append(np.append(uXMin, _uFixedT), uXMax)
        self.tIndex += 1

    def calculate(self,
                  logEvery: int = 1,
                  tToLogs: List[np.float64] = None,
                  noLogging: bool = False):
        if noLogging:
            self._calculateWithNoLogging()
        else:
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

    def _calculateWithNoLogging(self):
        for tIndex in range(self.tIndexMax):
            self.update()

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

    def plot(self, axes: plt.Axes, color: str, withLine=True):
        data = np.genfromtxt(self.fileName, delimiter=",")
        uTX = [[[], []] for t in self.tLogged]
        for point in data:
            uTX[self.tLogged.index(point[0])][0].append(point[1])
            uTX[self.tLogged.index(point[0])][1].append(point[2])

        for i, t in enumerate(self.tLogged):
            if withLine:
                axes.plot(uTX[i][0],
                          uTX[i][1],
                          marker=".",
                          markersize=0,
                          linewidth=1.0,
                          color=color,
                          label=str(t))
            else:
                axes.plot(uTX[i][0],
                          uTX[i][1],
                          marker=".",
                          markersize=2.0,
                          linewidth=0,
                          color=color,
                          label=str(t))

    def x(self, xIndex: int) -> np.float64:
        return self.xMin + self.xDelta * xIndex

    @property
    def t(self):
        return self.tMin + self.tDelta * self.tIndex

    @property
    def tNext(self):
        return self.tMin + self.tDelta * (self.tIndex + 1)