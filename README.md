# Encoder-Decoder-Image-Captioning
Project for the course Deep Learning 046211 (Technion)
Winter 2023

Video:

[YouTube](https://youtu.be/i8Cnas7QrMc) - https://youtu.be/HsJHZepSWHU
(in hebrew).


- [Encoder Decoder Image Captioning](#Encoder-Decoder-Image-Captioning)
  * [Background](#Background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Results](#results)
  * [References](#references)

## Background
Fill from report.


## Prerequisites
Full lists of requirements are in the requirements.txt file.
Require python version is 3.10.9.
To install the requirements run:
```pip install -r requirements.txt```


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|'data.py' | Data loader and additional scripts for the flickr8k datasets.|
|'models.py' | All the models used in the project (Transformer, LSTM, resnet50k).|
|'train.py' | Training script.|
|Example_Images | Folder with example images for the README.md file.|
|'LSTM_optuna.py' | Optuna hyperparameter tuning script for the LSTM model.|
|'Transformer_optuna.py' | Optuna hyperparameter tuning script for the Transformer model.|
|'Show_images.py' | Script for showing the images in the dataset, and results for Transformer model.|
|'Transformer_full.csv'| Results for the Transformer model during final training.|

## Displaying Results
In order to display example images for the Transformer / LSTM model, run the following command:
`python Show_images_and_results.py --model_class`
where model_class is either 'Transformer' or 'LSTM'.
## Training
In order to train the model one should clone the repository, select the model class (Transformer / LSTM)
and select the required hyperparameters in the script (the optimal hyperparameters we used are already in the script).

## References:
* https://www.kaggle.com/code/itaishufaro/flickr-30k-data-loader-preparation-pytorch/edit (Based data loader on this script).