# AUCMEDI - A Framework for Automated Classification of Medical Images

### Work in Progress!

This framework is currently under active development.  
Right now, it is possible to utilize the AUCMEDI framework as an high-level API for building state-of-the-art medical image classification pipelines.

Features that are already supported by AUCMEDI:
- Binary, multi-class and multi-label image classification
- Handling class imbalance through loss weighting
- Stratified iterative sampling like percentage split and k-fold cross-validation
- Standard preprocessing functions like Padding, Resizing, Cropping, Normalization
- Extensive real-time image augmentation
- Automated data loading and batch generation
- Data IO interfaces for csv and subdirectory encoded datasets
- Focal loss function
- Transfer Learning on ImageNet weights
- Large library of popular modern deep convolutional neural network architectures
- Ensemble Learning techniques like Inference Augmenting

The main reason this developed project is already publicly available is due to ensure the public reproducibility of our RIADD challenge participation.

Planed milestones and features are:
- Fully automated black box
- Integration of bagging and stacking pipelines for utilizing ensemble learning techniques
- Documentation
- Examples & Tutorials
- Continuous Integration (via GitHub or TravisCI)
- Explainable AI (XAI) via Grad-Cam
- Publication

Stay tuned and please have a look on AUCMEDI in a few month, again!

## Getting started: 60 seconds to automated medical image classification

```python
Examples coming soon :)
```

## Installation

There are two ways to install AUCMEDI:

- **Install AUCMEDI from PyPI (recommended):**

Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows or in a virtual environment without root, you will need to remove sudo to run the commands below.

```sh
pip install aucmedi
```

- **Alternatively: install AUCMEDI from the GitHub source:**

First, clone AUCMEDI using git:

```sh
git clone https://github.com/frankkramer-lab/aucmedi
```

Then, cd to the AUCMEDI folder and run the install command:

```sh
cd aucmedi
python setup.py install
```

## Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## How to cite / More information

Coming soon

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
