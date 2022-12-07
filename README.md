# Activity prediction with the help of neural networks

> Based on the InceptionTime model.
>
>Link to [repo](https://github.com/hfawaz/InceptionTime).
> Link to [paper](https://arxiv.org/pdf/1909.04939.pdf).

## Setup

- *env_files/* stores python environment
- Install a conda environment with *environment_cuda.yml* and a python virtual env with *requirements.txt*.
  - conda env create -f enf_files/environment_cuda.yml (only necessary if you want to train over the GPU)
  - pip install -r env_files/requirements.txt
- run nbdev_install_hooks to make sure notebooks metadata is cleaned before pushing to repo!

## Data

Two different Datasets:

1. in [input/separated](https://github.com/MinkTec/DataScience/tree/main/activities_neural_network/input/separated) are the csv files of [Armans student thesis](https://github.com/MinkTec/DataScience/tree/main/activities_neural_network/input/studienarbeit_arman.pdf).
The Dataset consists of 14 different activities, recorded by 30 persons.
    - Data has 50 sensor features (Sensor length: 25) and x,y,z acceleration.
2. The second dataset are the timeseries_data measurements directly saved to the cloud
    - downloaded to the repo from aws [aws_downloader.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/aws_downloader.ipynb)
    - aws needs to be installed for the command line over [this Link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
    - then you need to login to the MinkTec aws with *aws configure* and the ipi keys received from the AWS Master.
    - or directly used from the Nextcloud external data folder
    - Data has potentially variable sensor features, x,y,z acceleration and gyroscope data.

## Scripts and Directories

- [aws_downloader.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/aws_downloader.ipynb): Download the timeseries_data from aws and saves the data to disk at *input/timeseries_data*.
- [run_inception_tuner.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/run_inception_tuner.ipynb): Tweak model parameter and tune + fit an InceptionTime model.
- [create_models/](https://github.com/MinkTec/DataScience/tree/main/activities_neural_network/create_models): Scripts of inception_tuner model, Arman's models and Autokeras models.
- [utils/](https://github.com/MinkTec/DataScience/tree/main/activities_neural_network/utils): Helper scripts around data preprocessing, visualization, saving tflite models, etc.
- [input/](https://github.com/MinkTec/DataScience/tree/main/activities_neural_network/input): Datasets are stored here. Arman's old Dataset and new timeseries_data
- [run_models.ipynb DEPRECATED](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/run_models.ipynb): Runs Arman's old models and several Autokeras default models and compares them.
- [notebooks/ DEPRECATED](https://github.com/MinkTec/DataScience/tree/main/activities/neural/networks/notebooks) old Scripts

## Usage

### On Armans old dataset

- In [run_inception_tuner.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/run_inception_tuner.ipynb) use *load_armans_dataset* function and tweak hyperparameters.

### New measurements (AWS)

- run [aws_downloader.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/aws_downloader.ipynb) or connect to the Nextcloud external data source
- In [run_inception_tuner.ipynb](https://github.com/MinkTec/DataScience/blob/main/activities_neural_network/run_inception_tuner.ipynb) use *load_timeseries_dataset* function and tweak hyperparameters.
- CURRENTLY NOT ENOUGH DATA AND PARTICIPANTS TO TRAIN A SENSIBLE MODEL BUT WORKS IN THEORY

## Takeaways from working with Armans Dataset

- Prediction of 7 basic activities: ~ 93% acc and 92% f1 score.
- Not the full sensor length is important. ```n_keep = 12``` seems to be enough.
  - look out for this parameter later, might be necessary to increase if activities use higher back / neck or arms.
- Scaling of the data results in a worse model.
  - minmax scaling results in a garbage model.
  - standardization results in a slightly worse model (~ 2-5 % accuracy).
  - use Z-Score normalization for more sensible feature importance
- the worst predictions are on the walkingUpstairs category.
  - might be because of the low amount of data available.
- PCA with 15-10 components seems to work the best (might even improve the result a little bit).
- FocalLoss (for skewed classes) performs a bit worse.
- F1-Loss performs the best. especially on walkingUpstairs category -> higher f1-score

## Takeaways from newly created Dataset

- only use data starting at 19.10.22 (prior were sometimes accuracy values wrong)

## TODO

- Create sensible train, validation and test sets. Which split?
  - Smartphone model?
  - Sensor model?
- Make sure dataset does not get skewed (1 Activity or Person dominates the data)
  - visualize every persons contribution, for every activity
- Recognize movement and remove data during standstill (Waiting at traffic lights during cycling)
  - right now done by std over acc x and z over some threshold
    - good enough?
- Try different models with the help of fastai and tsai
- Create a Sample Dataset to iterate on more easily (~5 - 20% of the data)
- Maybe recreate the dataset collection with 15 Hz
- Expand hyperparameter search space
- learn differences between the tflite models
  - which model improves performance?
  - is the model size essential
  - inference speed
  - raw and quant model are the fastest right now
- Potential Cloud GPU Providers:
  - [Runpod](https://www.runpod.io/)
  - [VastAI](https://vast.ai/)
  - [Paperspace](https://www.paperspace.com/)
