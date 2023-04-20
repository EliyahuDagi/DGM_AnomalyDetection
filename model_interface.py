import abc
from torch.utils.data import Dataset


class GenerativeModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sample(self, n):
        pass
    
    @abc.abstractmethod
    def likelihood(self, x, pad):
        pass

    @abc.abstractmethod
    def criterion(self):
        pass


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, cfg):
        pass

    @abc.abstractmethod
    def prepare(self, model: GenerativeModel, pad, train_set: Dataset, train_pad: int):
        pass

    @abc.abstractmethod
    def predict_anomaly_score(self, data) -> float:
        pass
