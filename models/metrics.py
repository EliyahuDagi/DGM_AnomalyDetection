import torch
import tqdm

from model_interface import Metric, GenerativeModel
import numpy as np
from torch.utils.data import Dataset, DataLoader


class LikelihoodMetric(Metric):
    def __init__(self):
        self.model = None
        return

    def prepare(self, model: GenerativeModel, train_set: Dataset):
        self.model = model

    def predict_anomaly_score(self, data):
        # use the ELBO as approximation for the likelihood
        return self.model.likelihood(data).tolist()


class TypicalityMetric(Metric):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.model = None
        self.Hn = 0

    def prepare(self, model: GenerativeModel, train_set: Dataset):
        self.model = model
        # estimate entropy by re-substitution estimator:
        # -1 / m * sum(log_p)
        # using the training data as described https://arxiv.org/pdf/1906.02994.pdf eq. 5
        count_samples = 0
        sum_log_p = 0
        train_loader = DataLoader(train_set, batch_size=16, shuffle=False, num_workers=0)
        for train_batch, _ in tqdm.tqdm(train_loader):
            batch_log_p = self.model.likelihood(train_batch)
            sum_log_p += np.sum(batch_log_p)
            count_samples += batch_log_p.shape[0]
        self.Hn = -sum_log_p / count_samples

    def typicality_test(self, batch_likelihood):
        batch_info = np.sum(batch_likelihood) / -batch_likelihood.shape[0]
        return np.abs(batch_info - self.Hn)

    def predict_anomaly_score(self, data) -> float:
        scores = []
        likelihood = self.model.likelihood(data)
        num_batches = len(likelihood) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            cur_batch_likelihood = likelihood[start_idx: start_idx + self.batch_size]
            score = self.typicality_test(cur_batch_likelihood)
            scores.append(-score)
        return scores
