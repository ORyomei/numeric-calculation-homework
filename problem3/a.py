from scipy import optimize
import cmath


def at(func, theta):
    t = [(1, 2), ]
    t.append((optimize.fsolve(func, 5, args=(theta,))[0], 34))
    print(t)


def stableAreaRungeKutta2(r: float, theta: float) -> float:
    z = cmath.rect(r, theta) - 0.5
    Z = 1 + z + z ** 2 / 2
    return (Z * Z.conjugate() - 1).real


at(stableAreaRungeKutta2, 2)
