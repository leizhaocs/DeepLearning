import random

# replay memory
class ReplayMemory():

    # constructor
    def __init__(self, capacity):

        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    # store a new experience into the memory
    def push(self, experience):

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count%self.capacity] = experience

        self.push_count += 1

    # randomly sample a batch of experiences
    def sample(self, batch_size):

        return random.sample(self.memory, batch_size)

    # can only sample if there is enough experiences
    def can_provide_sample(self, batch_size):

        return len(self.memory) >= batch_size
