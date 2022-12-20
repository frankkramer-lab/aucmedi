![aucmedi_logo](docs/images/aucmedi.logo.description.png)

[![shield_python](https://img.shields.io/pypi/pyversions/aucmedi?style=flat-square)](https://www.python.org/)
[![shield_build](https://img.shields.io/github/actions/workflow/status/frankkramer-lab/aucmedi/build-package.yml?branch=master&style=flat-square)](https://github.com/frankkramer-lab/aucmedi)
[![shield_coverage](https://img.shields.io/codecov/c/gh/frankkramer-lab/aucmedi?style=flat-square)](https://app.codecov.io/gh/frankkramer-lab/aucmedi/)
[![shield_docs](https://img.shields.io/website?down_message=offline&label=docs&style=flat-square&up_message=online&url=https%3A%2F%2Ffrankkramer-lab.github.io%2Faucmedi%2Freference%2F)](https://frankkramer-lab.github.io/aucmedi/reference/)
[![shield_pypi_version](https://img.shields.io/pypi/v/aucmedi?style=flat-square)](https://pypi.org/project/aucmedi/)
[![shield_pypi_downloads](https://img.shields.io/pypi/dm/aucmedi?style=flat-square)](https://pypistats.org/packages/aucmedi)
[![shield_license](https://img.shields.io/github/license/frankkramer-lab/aucmedi?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0.en.html)

The open-source software AUCMEDI allows fast setup of medical image classification pipelines with state-of-the-art methods via an intuitive, high-level Python API or via an AutoML deployment through Docker/CLI.

**AUCMEDI provides several core features:**  
- Wide range of 2D/3D data entry options with interfaces to the most common medical image formats such as DICOM, MetaImage, NifTI, PNG or TIF already supplied.
- Selection of pre-processing methods for preparing images, such as augmentation processes, color conversions, windowing, filtering, resizing and normalization.
- Use of deep neural networks for binary, multi-class as well as multi-label classification and efficient methods against class imbalances using modern loss functions such as focal loss.
- Library from modern architectures, like ResNet up to EfficientNet and Vision-Transformers (ViT)⁠.
- Complex ensemble learning techniques (combination of predictions) using test-time augmentation, bagging via cross-validation or stacking via logistic regressions.
- Explainable AI to explain opaque decision-making processes of the models using activation maps such as Grad-CAM or backpropagation.
- Automated Machine Learning (AutoML) mentality to ensure easy deployment, integration and maintenance of complex medical image classification pipelines (Docker).

## Resources
- Website: [AUCMEDI Website - Home](https://frankkramer-lab.github.io/aucmedi/)
- Git Repository: [GitHub - frankkramer-lab/aucmedi](https://github.com/frankkramer-lab/aucmedi)
- Documentation: [AUCMEDI Wiki - API Reference](https://frankkramer-lab.github.io/aucmedi/reference/)
- Getting Started: [AUCMEDI Website - Getting Started](https://frankkramer-lab.github.io/aucmedi/getstarted/intro/)
- Examples: [AUCMEDI Wiki - Examples](https://frankkramer-lab.github.io/aucmedi/examples/framework/)
- Tutorials: [AUCMEDI Wiki - Tutorials](https://frankkramer-lab.github.io/aucmedi/examples/tutorials/)
- Applications: [AUCMEDI Wiki - Applications](https://frankkramer-lab.github.io/aucmedi/examples/applications/)
- PyPI Package: [PyPI - aucmedi](https://pypi.org/project/aucmedi/)
- Docker Image: [GitHub - ghcr.io/frankkramer-lab/aucmedi](https://github.com/frankkramer-lab/aucmedi/pkgs/container/aucmedi)
- Zenodo Repository: [Zenodo - AUCMEDI](https://zenodo.org/record/6633540)


## Getting started: 60 seconds to automated medical image classification

Simply install AUCMEDI with a single line of code via pip.

**Install AUCMEDI via PyPI**
```sh
pip install aucmedi
```

Now, you can build a state-of-the-art medical image classification pipeline via
the standardized AutoML interface or a custom pipeline with the framework interface.

### AutoML

**Train a model and classify unknown images**
```bash
# Run training with default arguments, but a specific architecture
aucmedi training --architecture "DenseNet121"

# Run prediction with default arguments
aucmedi prediction
```
### Framework

Your custom pipeline with just the 3 AUCMEDI pillars:
- Pillar #1: `input_interface()` for obtaining general dataset information
- Pillar #2: `NeuralNetwork()` for the deep learning model
- Pillar #3: `DataGenerator()` for a powerful interface to load any images/volumes into your model

**Build a pipeline**
```python
# AUCMEDI library
from aucmedi import *

# Pillar #1: Initialize input data reader
ds = input_interface(interface="csv",
                     path_imagedir="/home/muellerdo/COVdataset/ct_slides/",
                     path_data="/home/muellerdo/COVdataset/classes.csv",
                     ohe=False,           # OHE short for one-hot encoding
                     col_sample="ID", col_class="PCRpositive")
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Pillar #2: Initialize a DenseNet121 model with ImageNet weights
model = NeuralNetwork(n_labels=nclasses, channels=3,
                       architecture="2D.DenseNet121",
                       pretrained_weights=True)
```

**Train a model and use it!**
```python
# Pillar #3: Initialize training Data Generator for first 1000 samples
train_gen = DataGenerator(samples=index_list[:1000],
                          path_imagedir="/home/muellerdo/COVdataset/ct_slides/",
                          labels=class_ohe[:1000],
                          image_format=image_format,
                          resize=model.meta_input,
                          standardize_mode=model.meta_standardize)
# Run model training with Transfer Learning
model.train(train_gen, epochs=20, transfer_learning=True)

# Pillar #3: Initialize testing Data Generator for 500 samples
test_gen = DataGenerator(samples=index_list[1000:1500],
                         path_imagedir="/home/muellerdo/COVdataset/ct_slides/",
                         labels=None,
                         image_format=image_format,
                         resize=model.meta_input,
                         standardize_mode=model.meta_standardize)
# Run model inference for unknown samples
preds = model.predict(test_gen)

# preds <-> NumPy array with shape (500,2)
# -> 500 predictions with softmax probabilities for our 2 classes
```

## How to cite / More information

AUCMEDI is currently unpublished. But coming soon!

In the meantime:  
Please cite our application manuscript as well as the AUCMEDI GitHub repository:

```
Mayer, S., Müller, D., & Kramer F. (2022). Standardized Medical Image Classification
across Medical Disciplines. [Preprint] https://arxiv.org/abs/2210.11091.

@article{AUCMEDIapplicationMUELLER2022,
  title={Standardized Medical Image Classification across Medical Disciplines},
  author={Simone Mayer, Dominik Müller, Frank Kramer},
  year={2022}
  eprint={2210.11091},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

```
Müller, D., Mayer, S., Hartmann, D., Schneider, P., Soto-Rey, I., & Kramer, F. (2022).
AUCMEDI: a framework for Automated Classification of Medical Images (Version X.Y.Z) [Computer software].
https://doi.org/10.5281/zenodo.6633540. GitHub repository. https://github.com/frankkramer-lab/aucmedi
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
