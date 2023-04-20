# AnomalyDetectionDGM
## Table Of Contents
* Purpose
* Project structure
* Classes and Interfaces
* Getting Started

## Purpose
Using Deep Generative models (VAE, real NVP) for detecting out of distribution images
## Project Structure
run.py - the main script which you should run
model_interface.py specify 2 main interfaces GenerativeModel and Metric.
cfg.yaml - configuration for run.py(which datasets, models or metrics to use)
train.py - trains the generative model
* models\ - GenerativeModel and Metric interfaces implementations.
* results\ - output directory save model, samples and graphs. 
GenrativeModel allow you to sample using the model and to compute the likelihood(or lower bound for that in VAE).
Metric should use the model liklihood to separate between trained and un trained data.
You should run: run.py to run this project.
run wil use the cfg.yaml to know which model, metric and dataset to load
It will run twice once for original dataset and after that for features dataset.
## Getting Started
1. Installation process
   * git clone https://github.com/EliyahuDagi/DGM_AnomalyDetection.git
2. Software dependencies
   * pip install -r requirements.txt






