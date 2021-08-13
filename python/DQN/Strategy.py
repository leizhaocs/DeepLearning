import math

# epsilon greedy function
class EpsilonGreedyStrategy():

    # constructor
    def __init__(self, start, end, decay):

        self.start = start
        self.end = end
        self.decay = decay

    # the the current epsilon value
    def get_exploration_rate(self, current_step):

        return self.end + (self.start-self.end) * math.exp(-1. * current_step * self.decay)
