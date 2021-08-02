from pyqtgraph import GraphicsWindow, mkPen, PlotItem, PlotDataItem
from pyqtgraph.Qt import QtCore, QtGui
import sys
from threading import Thread

from numpy import (cos, float64, matrix, ndarray, power, sin, zeros, block,
                   eye, array)
from numpy.linalg import solve
from scipy.linalg import solve_continuous_are
from numeric_calculation import NumericCalculation

from init_prob import (l1, l2, m2, mu, g, theta12Initial, theta12_DInitial,
                       tDelta, logEveryT, torqueNoise)


class DoublePendulumGraph:
    win: GraphicsWindow
    nc: NumericCalculation
    plt: PlotItem
    line: PlotDataItem
    timer: QtCore.QTimer

    def __init__(self):
        self.setUI()
        self.setTimer()

    def setUI(self):
        self.win = GraphicsWindow()
        self.win.setWindowTitle('Controled double pendulum')
        self.win.resize(600, 600)
        self.plt = self.win.addPlot()
        graphsize = (l1 + l2) * 1.2
        self.plt.setYRange(-graphsize, graphsize)
        self.plt.setXRange(-graphsize, graphsize)
        self.line = self.plt.plot(pen=mkPen("w", width=3),
                                  symbol='o',
                                  symbolSize=14)

    def setTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(logEveryT * 1000)

    def setNumericCalculation(self, nc: NumericCalculation):
        self.nc = nc

    def ncUpdate(self):
        while True:
            self.nc.update()
            if self.nc._isToLog(0, logEveryT=logEveryT):
                break

    def render(self):
        t = self.nc.t
        theta1, theta2 = self.nc.x
        xy1 = l1 * array([sin(theta1), cos(theta1)])
        xy2 = xy1 + l2 * array([sin(theta2), cos(theta2)])

        xs = [0, xy1[0], xy2[0]]
        ys = [0, xy1[1], xy2[1]]
        self.line.setData(xs, ys)
        self.plt.setLabel("top", "t={}".format(round(t, 2)))

    def update(self):
        self.ncUpdate()
        self.render()


def noise():
    global gNoised
    while True:
        input()
        print("noised", end="")
        gNoised = True


O2 = matrix(zeros((2, 2)), dtype=float64)
I1 = matrix(eye(1), dtype=float64)
I2 = matrix(eye(2), dtype=float64)
I4 = matrix(eye(4), dtype=float64)

V = matrix([
    [mu * l1, l2],
    [l1, l2],
], dtype=float64)
A_ = g * V.I * matrix([[mu, 0], [0, 1]])
B_ = V.I * matrix([[1 / l1], [0]]) / m2

A = block([[O2, I2], [A_, O2]])
B = block([[float64(0)], [float64(0)], [B_]])
Q = I4
R = I1

X = solve_continuous_are(A, B, Q, R)
K = R.I * B.T * X

gNoised = False


def coefficientVector(theta12: ndarray, theta12_D: ndarray) -> ndarray:
    global gNoised
    theta1, theta2 = theta12
    theta1_D, theta2_D = theta12_D
    deltaTheta = theta2 - theta1
    theta = matrix(block([theta12, theta12_D])).T
    vector = zeros(2, dtype=float64)
    # tau1 = (-K * theta)[0, 0] + gNoised * torqueNoise
    tau1 = (-K * theta)[0, 0]
    vector[0] = l2 * power(theta2_D, 2) * sin(deltaTheta) + mu * g * sin(
        theta1) + tau1 / (l1 * m2)
    vector[1] = -l1 * power(theta1_D, 2) * sin(deltaTheta) + g * sin(
        theta2) + gNoised * torqueNoise / (l2 * m2)
    gNoised = False
    return vector


def coefficientMatrix(theta12: ndarray) -> matrix:
    theta1, theta2 = theta12
    deltaTheta = theta2 - theta1
    matrix_ = zeros((2, 2), dtype=float64)
    matrix_[0, 0] = mu * l1
    matrix_[0, 1] = l2 * cos(deltaTheta)
    matrix_[1, 0] = l1 * cos(deltaTheta)
    matrix_[1, 1] = l2
    return matrix_


def theta12_DD(theta12: ndarray, theta12_D: ndarray, t: float64) -> ndarray:
    return solve(coefficientMatrix(theta12),
                 coefficientVector(theta12, theta12_D))


nc = NumericCalculation()
nc.setInitialValue(theta12Initial, theta12_DInitial, tDelta, float64(0),
                   float64(1000))
nc.setEquation(theta12_DD)
nc.setMethod("RungeKutta4")

if __name__ == "__main__":
    noising = Thread(target=noise)
    noising.setDaemon(True)
    noising.start()

    doublePendulumGraph = DoublePendulumGraph()
    doublePendulumGraph.setNumericCalculation(nc)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
