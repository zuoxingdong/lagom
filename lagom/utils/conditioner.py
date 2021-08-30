import sys


class Conditioner(object):
    def __init__(self, start=0, stop=sys.maxsize, step=1):
        self.start = start
        self.stop = stop
        self.step = step
        self.n = self.start
    
    def __call__(self, x):
        if x > self.stop:
            return False
        check = x >= self.n
        if check:
            while self.n <= x:
                self.n += self.step
        return check
