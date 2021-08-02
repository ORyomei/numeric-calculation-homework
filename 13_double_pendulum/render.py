from pyqtgraph import GraphicsWindow, mkPen, PlotItem, PlotDataItem
from pyqtgraph.Qt import QtCore, QtGui
from numpy import genfromtxt, delete, array, ndarray, sin, cos
import sys

from init_prob import l1, l2, logEveryT

dataFileName = "data_controled.csv"


class PlotGraph:
    win: GraphicsWindow
    plt: PlotItem
    line: PlotDataItem
    timer: QtCore.QTimer
    data: ndarray

    def __init__(self):
        self.setUI()
        self.data = genfromtxt(dataFileName, delimiter=",")
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

    def render(self, t, theta1, theta2):
        xy1 = l1 * array([sin(theta1), cos(theta1)])
        xy2 = xy1 + l2 * array([sin(theta2), cos(theta2)])

        xs = [0, xy1[0], xy2[0]]
        ys = [0, xy1[1], xy2[1]]
        self.line.setData(xs, ys)
        self.plt.setLabel("top", "t={}".format(round(t, 2)))

    def update(self):
        if len(self.data) > 0:
            t, theta1, theta2 = self.data[0]
            self.data = delete(self.data, 0, 0)
            self.render(t, theta1, theta2)


if __name__ == "__main__":
    graphWin = PlotGraph()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()