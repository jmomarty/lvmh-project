import numpy as np


class ThompsonSampler:
    def __init__(self, number_of_arms=2):
        self.number_of_arms = number_of_arms
        self.alphas = np.ones(self.number_of_arms)
        self.betas = np.ones(self.number_of_arms)

    def _sample_from_beta_dist(self):
        self.thetas = np.random.beta(self.alphas, self.betas)

    def choose_arm(self):
        self._sample_from_beta_dist()
        self.chosen_arm = np.argmax(self.thetas)

    def update_beta_dist(self):
        self.alphas[self.chosen_arm] += self.current_reward
        self.betas[self.chosen_arm] += 1 - self.current_reward

    def get_reward(self, current_reward):
        self.current_reward = current_reward
