# EECS 600: Final Project

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