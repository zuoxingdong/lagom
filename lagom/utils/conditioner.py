class IntervalConditioner(object):
    def __init__(self, interval, mode):
        self.interval = interval
        assert mode in ['accumulative', 'incremental']
        self.mode = mode
        self.counter = 0
        if mode == 'incremental':
            self.total_n = 0
        
    def __call__(self, n):
        assert n >= 0
        if n == 0:
            return True
        else:
            if self.mode == 'accumulative':
                check = n >= (self.counter+1)*self.interval
            elif self.mode == 'incremental':
                self.total_n += n
                check = self.total_n >= (self.counter+1)*self.interval
            if check:
                self.counter += 1
            return check


class NConditioner(IntervalConditioner):
    def __init__(self, max_n, num_conditions, mode):
        self.max_n = max_n
        self.num_conditions = num_conditions
        interval = max_n/num_conditions
        super().__init__(interval, mode)
        
    def __call__(self, n):
        if self.counter >= self.num_conditions:
            return False
        else:
            return super().__call__(n)
