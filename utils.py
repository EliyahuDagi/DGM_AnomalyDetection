from __future__ import annotations
import os
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Grayscale, Pad, Lambda, Normalize

from timm import create_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from model_interface import GenerativeModel, Metric
from models.metrics import LikelihoodMetric, TypicalityMetric
from models.VAE import VAE
from models.real_nvp import RealNvpWrapper
from datasets import TensorDataSet
from tqdm import tqdm


def get_data_set(dataset_name, dataset_dir, feature_extractor_name='') -> Tuple[Dataset, Dataset]:
    if 'features' not in dataset_name:
        os.makedirs(dataset_dir, exist_ok=True)
    if dataset_name == 'mnist':
        transforms = Compose([ToTensor(), Pad(2)])
        train_data = datasets.MNIST(root=dataset_dir, train=True, transform=transforms, download=True)
        test_data = datasets.MNIST(root=dataset_dir, train=False, transform=transforms, download=True)
    elif dataset_name == 'fashion_mnist':
        transforms = Compose([ToTensor(), Pad(2)])
        train_data = datasets.FashionMNIST(root=dataset_dir, train=True, transform=transforms, download=True)
        test_data = datasets.FashionMNIST(root=dataset_dir, train=False, transform=transforms, download=True)
    elif dataset_name == 'cifar':
        transforms = Compose([Grayscale(), ToTensor()])
        train_data = datasets.CIFAR10(root=dataset_dir, train=True, transform=transforms, download=True)
        test_data = datasets.CIFAR10(root=dataset_dir, train=False, transform=transforms, download=True)
    elif 'features' in dataset_name:
        ensure_features_datasets(dataset_name=dataset_name, model_name=feature_extractor_name,
                                 data_dir=dataset_dir)
        train_data = TensorDataSet(dataset_dir=os.path.join(dataset_dir, 'train'))
        test_data = TensorDataSet(dataset_dir=os.path.join(dataset_dir, 'test'))
    return train_data, test_data


def split_train_val(dataset: Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
    dataset_len = len(dataset)
    train_len = int(split_ratio * dataset_len)
    valid_len = dataset_len - train_len
    return torch.utils.data.random_split(dataset, [train_len, valid_len])


def load_data_sets(cfg, data_set_type, feature_extractor) -> Tuple[Dict, Dict]:
    train_cfg = cfg.train_params
    train_loaders = dict()
    test_loaders = dict()
    for dataset_name in cfg.datasets[data_set_type]:
        dataset_dir = os.path.join('data', dataset_name)
        train_data, test_data = get_data_set(dataset_name, dataset_dir, feature_extractor_name=feature_extractor)
        train_dataset, valid_dataset = split_train_val(train_data, cfg.train_params.train_val_ratio)
        train_loaders[dataset_name] = (train_dataset, valid_dataset)
        test_loaders[dataset_name] = DataLoader(test_data, batch_size=train_cfg.train_batch_size, shuffle=False, num_workers=train_cfg.num_workers)
    return train_loaders, test_loaders


def load_models(cfg, data_set_type) -> Dict:
    models = dict()
    if data_set_type == 'original':
        input_shape = cfg.image_shape
        preprocess = True
    else:
        input_shape = cfg.feature_shape
        preprocess = False
    in_channels = input_shape[0]
    for model_name, model_params in cfg.models.items():
        model_params.in_channels = in_channels
        model_params.input_shape = input_shape
        if model_name == 'VAE':
            model = VAE(**model_params, type=data_set_type)
        elif model_name == 'RealNVP':
            model = RealNvpWrapper(preprocess=preprocess, type=data_set_type, **model_params)
        models[model_name] = model
    return models


def load_metrics(cfg) -> Dict:
    metrics = dict()
    for metric_name, metric_params in cfg.metrics.items():
        if metric_name == 'Likelihood':
            metric = LikelihoodMetric(**metric_params)
        elif metric_name == 'Typicality':
            metric = TypicalityMetric(**metric_params)
        metrics[metric_name] = metric
    return metrics


def evaluate_anomaly_detect(cfg, model_name: str, model: GenerativeModel,
                            test_sets, train_sets, trained_dataset,
                            metric_name: str, metric: Metric):
    scores = []
    labels = []
    train_set = train_sets[trained_dataset][0]
    metric.prepare(model=model, train_set=train_set)
    for dataset_name, test_set in test_sets.items():
        data_set_scores = metric.predict_anomaly_score(data=test_set)
        scores += data_set_scores
        is_anomaly = trained_dataset != dataset_name
        labels += ([0 if is_anomaly else 1] * len(data_set_scores))
    return roc_info(y_predicted=scores, y_true=labels)


def roc_info(y_predicted, y_true):
    fpr, tpr, threshold = roc_curve(y_true, y_predicted)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    return roc_auc


def get_feature_extractor(model_name):
    return create_model(model_name,
                        pretrained=True,
                        in_chans=1,
                        num_classes=0,
                        global_pool='',
                        output_stride=4,
                        features_only=True,
                        out_indices=[2]
                        )


def ensure_features_datasets(dataset_name, model_name, data_dir):
    if os.path.isdir(data_dir):
        return
    os.makedirs(data_dir, exist_ok=True)
    base_name = dataset_name.split('-')[0]
    original_data_set_path = os.path.join(data_dir, os.pardir, base_name)

    train_set, test_set = get_data_set(base_name, original_data_set_path)
    feat_extractor = get_feature_extractor(model_name)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    create_dataset_features(train_set, phase='train', model=feat_extractor,
                            preprocess=None, data_dir=data_dir)
    create_dataset_features(test_set, phase='test', model=feat_extractor,
                            preprocess=None, data_dir=data_dir)


def create_dataset_features(dataset, phase, model, preprocess, data_dir):
    phase_path = os.path.join(data_dir, phase)
    os.makedirs(phase_path, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        sample_index = 0
        for x, y in tqdm(loader, desc=f'create features {os.path.basename(data_dir)}', total=len(loader)):
            # x = preprocess(x)
            features = model(x.to(device))[0].cpu()
            for i in range(features.shape[0]):
                out_path = os.path.join(phase_path, f'{sample_index}.pt')
                torch.save({'x': features[i],
                            'y': y[i]}, out_path)
                sample_index += 1


if __name__ == '__main__':
    ensure_features_datasets('mnist-features', 'mobilenetv3_rw', 'data')
    ensure_features_datasets('fashion_mnist-features', 'mobilenetv3_rw', 'data')
    ensure_features_datasets('cifar-features', 'mobilenetv3_rw', 'data')