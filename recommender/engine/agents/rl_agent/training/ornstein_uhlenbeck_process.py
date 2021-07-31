import numpy as np


class OrnsteinUhlenbeckProcess:
    def __init__(self, K, SE, mu=0, theta=0.15, sigma=0.2):
        self.action_dims = (K, SE)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dims) * self.mu

    def reset_states(self):
        self.X = np.ones(self.action_dims) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(*self.action_dims)
        self.X = self.X + dx
        return self.X
