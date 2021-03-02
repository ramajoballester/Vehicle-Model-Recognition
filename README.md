# Vehicle-Model-Recognition (v0.2)
Vehicle model detection using deep siamese networks

## Installation

Clone this repository in your workspace

```
git clone https://github.com/ramajoballester/Vehicle-Model-Recognition.git
```

Install the required python packages

```
cd Vehicle-Model-Recognition/
sudo -H pip install -r requirements.txt
```

Download and extract the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) inside ```Vehicle-Model-Recognition/```. The complete dataset can be downloaded directly from [here](http://imagenet.stanford.edu/internal/car196/car_ims.tgz) and the annotation file from [here](http://imagenet.stanford.edu/internal/car196/cars_annos.mat).

## Dataset preparation 

After extracting the dataset, prepare the dataset directory

```
python src/pydataset_transformation.py
```

## Network training

Run the script ```train.py```. The training can be customized with the following parameters:

- arch: network architecture. VGG16 implemented in 3 configurations [VGG16A, VGG16D, VGG16E].
- batch_size: batch size for training.
- epochs: number of epochs.
- lr: learning rate.
- loss: loss function [categorical_crossentropy, categorical_hinge, KLD, MSE].
- metrics: metrics for training visualization [categorical_accuracy].
- model: path to load the model from (if included).
- n_classes: number of car models to load for training.
- n_elements: number of car per model.
- optimizer: optimizer for loss reduction [Adam, SGD, RMS].
- resume: resume training flag


E.g., to train de VGG16D network with batch size of 16, 1000 epochs, learning rate of 1e-4, 50 car models and 100 elements per model:

```
python src/train.py -arch VGG16D -batch_size 16 -epochs 1000 -lr 1e-4 -n_classes 50 -n_elements 100
```
