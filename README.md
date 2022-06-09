![aucmedi_logo](docs/images/aucmedi.logo.description.png)

[![shield_python](https://img.shields.io/pypi/pyversions/aucmedi?style=flat-square)](https://www.python.org/)
[![shield_build](https://img.shields.io/github/workflow/status/frankkramer-lab/aucmedi/Python%20Package%20-%20Build?style=flat-square)](https://github.com/frankkramer-lab/aucmedi)
[![shield_coverage](https://img.shields.io/codecov/c/gh/frankkramer-lab/aucmedi?style=flat-square)](https://app.codecov.io/gh/frankkramer-lab/aucmedi/)
[![shield_docs](https://img.shields.io/website?down_message=offline&label=docs&style=flat-square&up_message=online&url=https%3A%2F%2Ffrankkramer-lab.github.io%2Faucmedi%2Freference%2F)](https://frankkramer-lab.github.io/aucmedi/reference/)
[![shield_pypi_version](https://img.shields.io/pypi/v/aucmedi?style=flat-square)](https://pypi.org/project/aucmedi/)
[![shield_pypi_downloads](https://img.shields.io/pypi/dm/aucmedi?style=flat-square)](https://pypistats.org/packages/aucmedi)
[![shield_license](https://img.shields.io/github/license/frankkramer-lab/aucmedi?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0.en.html)

The open-source software AUCMEDI allows fast setup of medical image classification pipelines with state-of-the-art methods via an intuitive, high-level Python API or via an AutoML deployment through Docker/CLI.

## Work in Progress!

This framework is currently under active development for publishing our first stable release.  

The main reason this developed project is already publicly available is due to get things rolling and ensure the reproducibility of our challenge participations, ongoing clinical studies as well as publications based on AUCMEDI.

**AUCMEDI is already fully functional by utilizing the Python API / framework.**

Right now, it is possible to utilize the AUCMEDI framework as an high-level API for building state-of-the-art medical image classification pipelines.  

But more things like CLI/Docker for AutoML and straightforward application are coming!  

**Stay tuned and please have a look on AUCMEDI in the end of July, again! :)**  

## Resources
- Website: [AUCMEDI Website - Home](https://frankkramer-lab.github.io/aucmedi/)
- Git Repository: [GitHub - frankkramer-lab/aucmedi](https://github.com/frankkramer-lab/aucmedi)
- Documentation: [AUCMEDI Wiki - API Reference](https://frankkramer-lab.github.io/aucmedi/reference/)
- Getting Started: [AUCMEDI Website - Getting Started](https://frankkramer-lab.github.io/aucmedi/intro/)
- Examples: [AUCMEDI Wiki - Examples](https://frankkramer-lab.github.io/aucmedi/examples/framework/)
- Tutorials: Coming soon.
- Applications: [AUCMEDI Wiki - Applications](https://frankkramer-lab.github.io/aucmedi/examples/applications/)
- PyPI Package: [PyPI - aucmedi](https://pypi.org/project/aucmedi/)
- Docker Hub: Coming soon.
- Zenodo Repository: [Zenodo - AUCMEDI](https://pypi.org/project/aucmedi/)

## Roadmap

**Features that are already supported by AUCMEDI:**
- [x] Binary, multi-class and multi-label image classification
- [x] Support for 2D as well as 3D data
- [x] Handling class imbalance through class weights & loss weighting like Focal loss
- [x] Stratified iterative sampling like percentage split and k-fold cross-validation
- [x] Standard preprocessing functions like Padding, Resizing, Cropping, Normalization
- [x] Extensive online image augmentation
- [x] Automated data loading and batch generation
- [x] Data IO interfaces for csv and subdirectory encoded datasets
- [x] Transfer Learning on ImageNet weights
- [x] Large library of popular modern deep convolutional neural network architectures
- [x] Ensemble Learning techniques like Inference Augmenting
- [x] Explainable AI (XAI) via Grad-Cam, Backpropagation, ...
- [x] Clean implementation of the state-of-the-art for competitive application like challenges
- [x] Full (and automatic) documentation of the complete API reference
- [x] Started creating examples & applications for the community
- [x] Available from PyPI for simple installation in various environments
- [x] Interface for metadata / pandas or NumPy table inclusion in model architectures
- [x] Unittesting -> CI/CD
- [x] Clean up Website
- [x] Integration of bagging and stacking pipelines for utilizing ensemble learning techniques
- [x] Integrate evaluation functions

**Planed milestones and features are:**
- [ ] Tutorials
- [ ] Support for AutoML via CLI and Docker
- [ ] Documentation for AutoML
- [ ] Publication

## Getting started: 60 seconds to automated medical image classification

Simply install AUCMEDI with a single line of code via pip.

**Install AUCMEDI via PyPI**
```sh
pip install aucmedi
```

Now, you can build a state-of-the-art and complex medical image classification pipeline with just the 3 AUCMEDI pillars.
- Pillar #1: `input_interface()` for obtaining general dataset information
- Pillar #2: `NeuralNetwork()` for the deep learning model
- Pillar #3: `DataGenerator()` for a powerful interface to load any images/volumes into your model

Let's build a COVID-19 Detection AI on CT scans!

**Build a pipeline**
```python
# AUCMEDI library
from aucmedi import *

# Pillar #1: Initialize input data reader
ds = input_interface(interface="csv",
                     path_imagedir="/home/muellerdo/COVdataset/ct_scans/",
                     path_data="/home/muellerdo/COVdataset/classes.csv",
                     ohe=False,           # OHE short for one-hot encoding
                     col_sample="ID", col_class="PCRpositive")
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Pillar #2: Initialize a DenseNet121 model with ImageNet weights
model = NeuralNetwork(n_labels=nclasses, channels=3,
                       architecture="2D.DenseNet121",
                       pretrained_weights=True)
```
Congratulations to your ready-to-use Medical Image Classification pipeline including data I/O, preprocessing and a deep learning based neural network model.

**Train a model and use it!**
```python
# Pillar #3: Initialize training Data Generator for first 1000 samples
train_gen = DataGenerator(samples=index_list[:1000],
                          path_imagedir="/home/muellerdo/COVdataset/ct_scans/",
                          labels=class_ohe[:1000],
                          image_format=image_format)
# Run model training with Transfer Learning
model.train(train_gen, epochs=20, transfer_learning=True)

# Pillar #3: Initialize testing Data Generator for 500 samples
test_gen = DataGenerator(samples=index_list[1000:1500],
                         path_imagedir="/home/muellerdo/COVdataset/ct_scans/",
                         labels=None,
                         image_format=image_format)
# Run model inference for unknown samples
preds = model.predict(test_gen)

# preds <-> NumPy array with shape (500,2)
# -> 500 predictions with softmax probabilities for our 2 classes
```

## How to cite / More information

AUCMEDI is currently unpublished. But coming soon!

In the meantime:  
Please cite our base framework MIScnn as well as the AUCMEDI GitHub repository:

```
Müller, D., Kramer, F. MIScnn: a framework for medical image segmentation with
convolutional neural networks and deep learning. BMC Med Imaging 21, 12 (2021).
https://doi.org/10.1186/s12880-020-00543-7
```

```
Müller, D., Mayer, S., Hartmann, D., Meyer, P., Schneider, P., Soto-Rey, I., & Kramer, F. (2022).
AUCMEDI: a framework for Automated Classification of Medical Images (Version 1.0.0) [Computer software].
GitHub repository. https://github.com/frankkramer-lab/aucmedi
```

Thank you for citing our work.

### Lead Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
