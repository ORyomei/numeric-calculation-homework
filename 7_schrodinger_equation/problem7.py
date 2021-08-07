from numeric_calculation import NumericCalculation
from numpy import float64, ndarray, power, array, cosh, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def V(t: float64):
    return -3.0 / power(cosh(t), 2)


def a(x: ndarray, t: float64, eigenValue: float64) -> ndarray:
    return (V(t) - eigenValue) * x


xInitial = array([1e-5], dtype=float64)
tInitial = -20.0
tFinal = 20.0
tDelta = 20 / 5000

gCalculator: NumericCalculation


def schrodingerEquation(eigenValue: float) -> float64:
    global gCalculator
    vInitial = xInitial * sqrt(V(tInitial) - eigenValue)

    calculator = NumericCalculation()
    calculator.setInitialValue(xInitial=xInitial,
                               vInitial=vInitial,
                               tDelta=tDelta,
                               tInitial=tInitial,
                               tFinal=tFinal)
    distanceInitial = calculator.distance

    def _a(x: ndarray, v: ndarray, t: float64) -> ndarray:
        return a(x, t, eigenValue)

    calculator.setEquation(_a)
    calculator.setMethod("RungeKutta4")
    calculator.calculate()
    gCalculator = calculator
    return calculator.distance - distanceInitial


figure = plt.figure(figsize=(16, 7))
axes = figure.add_subplot(111)
axes.set_xlabel("x")
axes.set_ylabel("u")

eigenValue = fsolve(schrodingerEquation, -0.5)
print(eigenValue)
gCalculator.normalize()
gCalculator.plot(axes, "b", label=r"$E={}$".format(eigenValue))

eigenValue = fsolve(schrodingerEquation, -2.0)
print(eigenValue)
gCalculator.normalize()
gCalculator.plot(axes, "g", label=r"$E={}$".format(eigenValue))

axes.legend()
plt.show()
