import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from thompson_sampling import ThompsonSampler


logging.basicConfig(level=logging.INFO)


def parse_rec_file(fn):
    with open(fn, 'r') as f:
        recs = [line for line in f]
    return recs


class ThompsonSamplingRecSys:
    def __init__(self, config_fn):
        self._load_config(config_fn)
        self.number_of_models = len(self.recs.keys())
        self.models = list(self.recs.keys())
        self.ts = ThompsonSampler(self.number_of_models)
        #self.plot()

    def _load_config(self, fn):
        with open(fn, 'r') as f:
            self.config = json.load(f)
        if 'rec_files' not in self.config:
            raise AssertionError('rec_files not in config file')
        self.rec_files = self.config['rec_files']
        if type(self.rec_files) != dict:
            raise TypeError('rec_files is not a dictionary')
        self._parse_rec_files()

    def _parse_rec_files(self):
        self.recs = {model: parse_rec_file(fn) for model, fn in self.rec_files.items()}

    def recommend(self):
        self.ts.choose_arm()
        self.chosen_model = self.models[self.ts.chosen_arm]
        if len(self.recs[self.chosen_model]) == 0:
            return 'No more recommandations in this model'
        return self.recs[self.chosen_model].pop(0)

    def get_feedback(self, feedback):
        self.ts.get_reward(feedback)
        self.ts.update_beta_dist()
        self.log_infos()
        #self.plot()

    def log_infos(self):
        for i, model in enumerate(self.models):
            logging.info(f'Updated Beta Distribution for model {model}:')
            logging.info(f'Alpha = {self.ts.alphas[i]}:')
            logging.info(f'Beta =  {self.ts.betas[i]}:')

    def plot(self):
        x = np.linspace(0, 1, 30)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, model in enumerate(self.models):
            ax.plot(x, beta.pdf(x, self.ts.alphas[i], self.ts.betas[i]), label=model)
        plt.legend(loc='lower left')
        plt.show()
