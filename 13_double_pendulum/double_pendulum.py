import matplotlib
from numeric_calculation import NumericCalculation
from init_prob import (l1, l2, mu, g, theta12Initial, theta12_DInitial, tDelta,
                       logEveryT)
from numpy import cos, float64, matrix, ndarray, power, sin, zeros
from numpy.linalg import solve
import matplotlib.pyplot as plt


def coefficientVector(theta12: ndarray, theta12_D: ndarray) -> ndarray:
    theta1, theta2 = theta12
    theta1_D, theta2_D = theta12_D
    deltaTheta = theta2 - theta1
    vector = zeros(2, dtype=float64)
    vector[0] = l2 * power(theta2_D,
                           2) * sin(deltaTheta) + mu * g * sin(theta1)
    vector[1] = -l1 * power(theta1_D, 2) * sin(deltaTheta) + g * sin(theta2)
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
                   float64(100))
nc.setEquation(theta12_DD)
nc.setMethod("RungeKutta4")
nc.setDataFileName("data.csv")
nc.calculate(logEveryT=logEveryT)

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$t$")

nc.plot(axes, "b", columns=[0, 1], label=r"$\theta_1$")
nc.plot(axes, "g", columns=[0, 2], label=r"$\theta_2$")

axes.legend()
plt.show()
