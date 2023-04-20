import os
from matplotlib import pyplot as plt
from train import train
from utils import load_data_sets, load_models, load_metrics, evaluate_anomaly_detect
from easydict import EasyDict
import yaml
import pandas as pd
import torch
from torchvision.utils import save_image, make_grid


def run(cfg, data_set_type):
    # load train_test loaders for each_dataset
    train_sets, test_sets = load_data_sets(cfg=cfg, data_set_type=data_set_type, feature_extractor=cfg.feature_extractor)
    # datasets len must be > 1
    assert len(train_sets) > 1
    assert len(test_sets) > 1
    # load anomaly metric(Likelihood, Typicality)
    metrics = load_metrics(cfg)
    # load models(VAE, RealNVP)
    models = load_models(cfg, data_set_type)
    result_df = pd.DataFrame(columns=['Model', 'Dataset', 'Metric', 'ROC_AUC'])

    # for each model train on 1 dataset and evaluate it against the rest using the anomaly metric
    for model_name, model in models.items():
        for dataset_name, train_dataset in train_sets.items():
            saved_model_path = os.path.join(cfg.debug.out_dir, model_name, dataset_name, 'model.pth')
            if cfg.train_params.use_saved_model and os.path.exists(saved_model_path):
                model.load_state_dict(torch.load(saved_model_path))
            else:
                train(model_name=model_name, model=model,
                      dataset_name=dataset_name, datasets=train_dataset,
                      cfg=cfg.train_params, output_dir=cfg.debug.out_dir)
                # load best model
                model.load_state_dict(torch.load(saved_model_path, map_location=model.device))

            model.eval()
            model = model.to(model.device)
            # sample images using the model
            if 'features' not in dataset_name:
                out_dir = os.path.join(cfg.debug.out_dir, model_name, dataset_name)
                samples_dir = os.path.join(out_dir, 'samples')
                with torch.no_grad():
                    samples = model.sample(20).cpu()
                    # samples = torch.sigmoid(samples)
                a, b = samples.min(), samples.max()
                samples = (samples - a) / (b - a + 1e-10)
                samples = samples.view(-1, 1, 32, 32)
                save_image(make_grid(samples), os.path.join(samples_dir, f'best_model_samples.png'))
            # for each metric calculate anomaly detection score
            # (AUC of ROC curve of the metric between trained and untrained datasets)

            for metric_name, metric in metrics.items():
                roc_auc = evaluate_anomaly_detect(cfg=cfg,
                                                  model_name=model_name, model=model,
                                                  test_sets=test_sets, train_sets=train_sets,
                                                  trained_dataset=dataset_name,
                                                  metric_name=metric_name, metric=metric)
                out_dir = os.path.join(cfg.debug.out_dir, model_name, dataset_name)
                plt.savefig(os.path.join(os.path.join(out_dir, f'{metric_name}_roc.jpg')))
                plt.close()
                result_df.loc[len(result_df.index)] = [model_name, dataset_name, metric_name, roc_auc]
            result_df.to_csv(os.path.join(cfg.debug.out_dir, f'summery_{data_set_type}.csv'), index=False)


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    for data_set_type in cfg.datasets.keys():
        run(cfg=cfg, data_set_type=data_set_type)

