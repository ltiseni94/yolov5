class GraspScore:
    def __init__(self,
                 conf_thres=0.5,
                 close_weight=0.35,
                 proximity_weight=0.4,
                 reset_counter_max=120):
        self.conf_thres = conf_thres
        self.previous_val = 0
        self.reset_counter_max = reset_counter_max
        self.reset_counter = 0
        self.cw = close_weight
        self.pw = proximity_weight
        self.mw = 1 - close_weight - proximity_weight
        assert 1 > self.mw > 0

    def __call__(self, close_prob, prox_score):
        self.reset_counter += 1
        if self.reset_counter == self.reset_counter_max:
            self.reset_counter = 0
            new_val = self.cw * close_prob + self.pw * prox_score
            new_val *= 1 / (self.cw + self.pw)
        else:
            new_val = self.mw * self.previous_val + self.cw * close_prob + self.pw * prox_score
        self.previous_val = new_val
        return new_val, new_val > self.conf_thres
