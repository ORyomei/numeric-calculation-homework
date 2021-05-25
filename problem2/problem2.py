from typing import Literal
import numpy as np
import matplotlib.pyplot as plt


class NumericCalculation:
    def setInitialValue(self,
                        xInitial: np.ndarray,
                        vInitial: np.ndarray,
                        tDelta: np.float64,
                        tInitial: float,
                        tFinal: float):
        self.xAndv = np.vstack([xInitial, vInitial])
        self.x1List = [xInitial[0]]
        self.x2List = [xInitial[1]]
        self.tDelta = tDelta
        self.t = tInitial
        self.tFinal = tFinal

    def setMethod(self, method: Literal["Euler", "RungeKutta2", "RungeKutta4", "Symplectic"]):
        if method == "Euler":
            self._update = self._updateEuler
        elif method == "RungeKutta2":
            self._update = self._updateRungeKutta2
        elif method == "RungeKutta4":
            self._update = self._updateRungeKutta4
        elif method == "Symplectic":
            self._update = self._updateSymplectic
        self.method = method

    def setEquation(self, a):
        self.a = a

    def _dxAnddv(self, xAndv: np.ndarray):
        return np.vstack([xAndv[1], self.a(xAndv[0])])

    def _updateEuler(self):
        k1 = self._dxAnddv(self.xAndv)
        self.xAndv += self.tDelta * k1
        self.x1List.append(self.xAndv[0][0])
        self.x2List.append(self.xAndv[0][1])
        self.t += self.tDelta

    def _updateRungeKutta2(self):
        k1 = self._dxAnddv(self.xAndv)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1)
        self.xAndv += self.tDelta * (k1 + k2) / 2
        self.x1List.append(self.xAndv[0][0])
        self.x2List.append(self.xAndv[0][1])
        self.t += self.tDelta

    def _updateRungeKutta4(self):
        k1 = self._dxAnddv(self.xAndv)
        k2 = self._dxAnddv(self.xAndv + self.tDelta * k1 / 2)
        k3 = self._dxAnddv(self.xAndv + self.tDelta * k2 / 2)
        k4 = self._dxAnddv(self.xAndv + self.tDelta * k3)
        self.xAndv += self.tDelta * (k1 + k2 * 2 + k3 * 2 + k4) / 6
        self.x1List.append(self.xAndv[0][0])
        self.x2List.append(self.xAndv[0][1])
        self.t += self.tDelta

    def _updateSymplectic(self):
        self.xAndv[0] += self.tDelta * self.xAndv[1]
        self.xAndv[1] += self.tDelta * self.a(self.xAndv[0])
        self.x1List.append(self.xAndv[0][0])
        self.x2List.append(self.xAndv[0][1])
        self.t += self.tDelta

    def plot(self, axes: plt.Axes, color: str):
        axes.scatter(self.x1List, self.x2List, marker='.',
                     color=color, label=self.method)

    def start(self) -> np.ndarray:
        while True:
            self._update()
            if self.t >= self.tFinal:
                break

    @property
    def x(self) -> np.ndarray:
        return self.xAndv[0]


def a(x: np.ndarray) -> np.ndarray:
    return -x / np.power(np.linalg.norm(x, ord=2), 3)


figure = plt.figure()
axes = figure.add_subplot(111)
axes.set_aspect('equal', adjustable='box')

euler = NumericCalculation()
euler.setInitialValue(xInitial=np.array([1, 0], dtype=np.float64),
                      vInitial=np.array([0, 1], dtype=np.float64),
                      tDelta=0.2,
                      tInitial=0.0,
                      tFinal=20.0)
euler.setMethod("Euler")
euler.setEquation(a)
euler.start()
euler.plot(axes, "b")

rungeKutta2 = NumericCalculation()
rungeKutta2.setInitialValue(xInitial=np.array([1, 0], dtype=np.float64),
                            vInitial=np.array([0, 1], dtype=np.float64),
                            tDelta=0.2,
                            tInitial=0.0,
                            tFinal=20.0)
rungeKutta2.setMethod("RungeKutta2")
rungeKutta2.setEquation(a)
rungeKutta2.start()
rungeKutta2.plot(axes, "r")

rungeKutta4 = NumericCalculation()
rungeKutta4.setInitialValue(xInitial=np.array([1, 0], dtype=np.float64),
                            vInitial=np.array([0, 1], dtype=np.float64),
                            tDelta=0.2,
                            tInitial=0.0,
                            tFinal=20.0)
rungeKutta4.setMethod("RungeKutta4")
rungeKutta4.setEquation(a)
rungeKutta4.start()
rungeKutta4.plot(axes, "g")

symplectic = NumericCalculation()
symplectic.setInitialValue(xInitial=np.array([1, 0], dtype=np.float64),
                           vInitial=np.array([0, 1], dtype=np.float64),
                           tDelta=0.2,
                           tInitial=0.0,
                           tFinal=20.0)
symplectic.setMethod("Symplectic")
symplectic.setEquation(a)
symplectic.start()
symplectic.plot(axes, "y")

axes.legend()
# figure.savefig("plot.png")
plt.show()
