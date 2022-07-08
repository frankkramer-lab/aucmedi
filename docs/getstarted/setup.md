## Prerequisite

AUCMEDI is tested and supported on the following systems:

- 64-bit Ubuntu 16.04 or later  
- Python 3.8 - 3.10  

AUCMEDI is heavily based on TensorFlow 2: an approachable, highly-productive interface for solving machine learning problems, with a focus on modern deep learning. It provides essential abstractions and building blocks for developing and shipping machine learning solutions with high iteration velocity.

In order to install AUCMEDI, verify that all requirements are complied and functional.


## AUCMEDI Installation

There are three ways to install AUCMEDI which depends on the preferred usage:

### 1) Install from PyPI (recommended)

Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows or in a virtual environment without root, you will need to remove sudo to run the commands below.

This will allow utilizing the framework (in your favorite Python IDE or in a Jupyter Notebook) or the AutoML module via the command line interface (CLI).

```sh
pip install aucmedi
```

### 2) Install from GitHub Container Registry

This will allow utilizing the AutoML module via Docker.

```sh
docker pull ghcr.io/frankkramer-lab/aucmedi:latest
```

### 3) Install from Source Code

First, clone AUCMEDI using git:

```sh
git clone https://github.com/frankkramer-lab/aucmedi
```

Then, cd to the 'aucmedi' folder and run the install command:

```sh title="Installation with setup.py (framework & AutoML)"
cd aucmedi
python setup.py install
```

```sh title="Installation with pip (framework & AutoML)"
cd aucmedi
pip install .
```

```sh title="Installation with Docker (AutoML)"
cd aucmedi
docker build -t ghcr.io/frankkramer-lab/aucmedi:latest .
```
