# Prerequisite

AUCMEDI is tested and supported on the following 64-bit system: Ubuntu 16.04 or later

As core langauge
Python

AUCMEDI is heavily based on TensorFlow 2: an approachable, highly-productive interface for solving machine learning problems, with a focus on modern deep learning. It provides essential abstractions and building blocks for developing and shipping machine learning solutions with high iteration velocity.

In order to run AUCMEDI



# AUCMEDI Installation

There are three ways to install AUCMEDI which depends on the preferred usage:

## Install from PyPI (recommended)

Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows or in a virtual environment without root, you will need to remove sudo to run the commands below.

This will allow utilizing the framework (in your favorite Python IDE or in a Jupyter Notebook) or the AutoML module via the command line interface (CLI).

```sh
pip install aucmedi
```

## Install from DockerHub for AutoML

Work in Progress. Coming soon.

## Alternatively: Install from the GitHub source

First, clone AUCMEDI using git:

```sh
git clone https://github.com/frankkramer-lab/aucmedi
```

Then, cd to the 'aucmedi' folder and run the install command:

```sh
cd aucmedi
python setup.py install
```
