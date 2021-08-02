from numpy import float64, array, pi

l1 = float64(0.5)
l2 = float64(0.2)
m1 = float64(0.01)
m2 = float64(0.05)
mu = 1 + m1 / m2
g = float64(9.8)

theta12Initial = array([-0.4, 0.4], dtype=float64)
theta12_DInitial = array([0, 0], dtype=float64)
tDelta = float64(0.001)
logEveryT = float64(1) / 60

torqueNoise = float64(15)
sigma1 = 1
sigma2 = 0.2