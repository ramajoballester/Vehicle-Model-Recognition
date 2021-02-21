# Vehicle-Model-Recognition (v0.1)
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

Download the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) inside ```Vehicle-Model-Recognition/```. The complete dataset can be downloaded directly from [here](http://imagenet.stanford.edu/internal/car196/car_ims.tgz) and the annotation file from [here](http://imagenet.stanford.edu/internal/car196/cars_annos.mat).

## Dataset preparation 

After downloading and extracting the dataset, run the pydataset_transformation.py Python script to create the dataset directory.

## Siamese network training

Run the train_cars.ipynb Jupyter Notebook to train the network. Still under development

## Python version

Run the pytrain_cars_classification.py script to train the network.
