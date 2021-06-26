# Vehicle-Model-Recognition (v0.3)
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

Download and extract the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) inside ```Vehicle-Model-Recognition/```. The complete dataset can be downloaded directly from [here](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz) and the annotation file from [here](http://ai.stanford.edu/~jkrause/car196/cars_annos.mat).

## Dataset preparation 

After extracting the dataset, prepare the dataset directory

```
python src/pydataset_transformation.py
```

## Network training parameters

Run the script ```train.py```. The training can be customized with the following parameters:

- arch: network architecture. VGG16 implemented in 3 configurations [VGG16A, VGG16D, VGG16E].
- batch_size: batch size for training.
- data_cfg: data labels configuration file.
- data_augmentation: data augmentation option.
- epochs: number of epochs.
- lr: learning rate.
- ls: label smoothing.
- loss: loss function [binary_crossentropy, categorical_crossentropy, categorical_hinge, KLD, MSE].
- metrics: metrics for training visualization [binary_accuracy, categorical_accuracy].
- model: path to load the model from (if included).
- multi_gpu: use all available GPUs for training.
- n_classes: number of car models to load for training.
- n_elements: number of car per model.
- optimizer: optimizer for loss reduction [Adam, SGD, RMS].
- output: Network output [classification, siamese].
- resume: resume training flag.
- train_cfg: load training configuration.


## Classification training

To train the VGG16D network for classification with batch size of 16, 1000 epochs, learning rate of 1e-4, 50 car models and 100 elements per model:

```
python src/train.py -arch VGG16D -batch_size 16 -epochs 1000 -lr 1e-4 -n_classes 50 -n_elements 100
```

## Siamese training

To train the siamese architecture, binary equivalents must be included. That means the loss must be binary_crossentropy and metrics should be binary_accuracy. E.g:

```
python train.py -arch VGG16_pretrained -batch_size 4 -epochs 5000 -lr 1e-5 -ls 0.2 -loss binary_crossentropy -metrics binary_accuracy -n_classes 5 -n_elements 300 -output siamese
```
