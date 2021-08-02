from numeric_calculation import NumericCalculation
from init_prob import l1, l2, m2, mu, g, theta12Initial, theta12_DInitial, tDelta, logEveryT, sigma1, sigma2
from numpy import cos, float64, matrix, ndarray, power, sin, zeros, block, eye, random
from numpy.linalg import solve
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

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


def coefficientVector(theta12: ndarray, theta12_D: ndarray) -> ndarray:
    theta1, theta2 = theta12
    theta1_D, theta2_D = theta12_D
    deltaTheta = theta2 - theta1
    theta = matrix(block([theta12, theta12_D])).T
    vector = zeros(2, dtype=float64)
    tau1 = (-K * theta)[0, 0] + random.normal(0, sigma1)
    tau2 = random.normal(0, sigma2)
    vector[0] = l2 * power(theta2_D, 2) * sin(deltaTheta) + mu * g * sin(
        theta1) + tau1 / (l1 * m2)
    vector[1] = -l1 * power(
        theta1_D, 2) * sin(deltaTheta) + g * sin(theta2) + tau2 / (l2 * m2)
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
                   float64(30))
nc.setEquation(theta12_DD)
nc.setMethod("RungeKutta4")
nc.setDataFileName("data_noised.csv")
nc.calculate(logEveryT=logEveryT)

figure = plt.figure(figsize=(14, 7))
axes = figure.add_subplot(111, xlabel="$t$")

nc.plot(axes, "b", columns=[0, 1], label=r"$\theta_1$")
nc.plot(axes, "g", columns=[0, 2], label=r"$\theta_2$")

axes.legend()
plt.show()