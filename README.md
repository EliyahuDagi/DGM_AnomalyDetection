# AnomalyDetectionDGM
## Table Of Contents
* Purpose
* Project structure
* Classes and Interfaces
* Important notes
* Getting Started

## Purpose
Using Deep Generative models (VAE, real NVP) for detecting out of distribution images
## Project Structure
run.py - the main script which you should run.\
model_interface.py specify 2 main interfaces GenerativeModel and Metric.\
cfg.yaml - configuration for run.py(which datasets, models or metrics to use).\
train.py - trains the generative model.\
data\ - store datasets ( such as mnist, cifar, etc.)\
models\ - GenerativeModel and Metric interfaces implementations.\
results\ - output directory saved model, images samples and graphs.
## Classes and Interfaces
*GenrativeModel* - allows you to sample using the model and to compute the likelihood(or lower bound for that in VAE).\

*Metric* - should use the model likelihood to separate between trained and un trained data.\
You should run: run.py to use this project.\
run wil use the cfg.yaml to know which model, metric and dataset to load.\
It will run twice once for original dataset and after that for features dataset.\
## Important Notes
* Each dataset will be downloaded into data/ directory\
* Notice that features dataset is computed one time and saved on the disk, so it need some space.\
   if you want to ignore it remove it from cfg.yaml "datasets".\
* if cfg['train_params']['use_saved_model'] is True model that exist in result will not be trained,
set it to false if you want to force training


## Getting Started
1. Installation process\
   git clone:  https://github.com/EliyahuDagi/DGM_AnomalyDetection.git
2. Software dependencies\
   pip install -r requirements.txt
3. Run: run.py






