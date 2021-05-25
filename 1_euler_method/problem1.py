import math


class Euler:
    y: float
    intervalTime: float

    def __init__(self, steps: int):
        self.steps = steps
        self.intervalTime = 1.0 / steps
        self.y = 1

    def _update(self):
        self.y = self.y + self.intervalTime * math.exp(-self.y)

    def finalValue(self):
        for _ in range(self.steps):
            self._update()
        return self.y


print("interval time, error / interval time")
for i in range(5):
    error = Euler(10 ** (i + 1)).finalValue() - math.log(1 + math.e)
    errorInterval = error * (10 ** (i + 1))
    print("10^-" + str(i + 1) + ", " + str(errorInterval))
