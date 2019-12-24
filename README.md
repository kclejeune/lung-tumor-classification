# EECS 600: Final Project

## Abstract

Lung tumor identification and classification is a challenging task which typically requires a trained medical professional to choose the best slices of a scan and accurately classify the chosen slices. With medical data privacy restrictions and regulations, it is difficult to collect sufficient data to construct a typical Convolutional Neural Network (CNN) to choose and classify Digital Imaging and Communications in Medicine (DICOM) slices. We propose an inductive transfer learning approach which applies hidden layer image representations from a residual neural network to our lung nodule classifier to classify groups of slices and provide recommendations to a user as to what groups may contain possible benign or malignant nodules.

## Install Requirements:

To install the requirements run:

```
pip install --user -r requirements.txt
```

## Fetching LIDC Data:
All the LIDC data can be found [here](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

Setting up PyLIDC requires a file called ".pylidcrc" to be placed in the users home directory with the following content:

```
[dicom]
path = PATH_TO_LIDC-IDRI_DATA
warn = True
```

The data is roughly 125gb and can be processed by running:

```
python lidc_parser.py
```

## Running Training:

Once the data is processed, to train classifier, run:

```
python transfer.py
```

The settings in this file under "constant values" can be modified to train the model to specific parameters
