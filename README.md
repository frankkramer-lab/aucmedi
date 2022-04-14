# AUCMEDI - a framework for Automated Classification of Medical Images

## Work in Progress!

This framework is currently under active development.  
The main reason this developed project is already publicly available is due to get things rolling and ensure the reproducibility of our challenge participations as well as publications based on AUCMEDI.

Right now, it is possible to utilize the AUCMEDI framework as an high-level API for building state-of-the-art medical image classification pipelines.  
But more things like CLI/Docker for AutoML and straightforward application are coming!  

**Stay tuned and please have a look on AUCMEDI in 1-2 month, again! :)**  

## Resources
- Website: [https://frankkramer-lab.github.io/aucmedi/](https://frankkramer-lab.github.io/aucmedi/)
- Git Repository: [https://github.com/frankkramer-lab/aucmedi](https://github.com/frankkramer-lab/aucmedi)
- API Documentation: [https://frankkramer-lab.github.io/aucmedi/reference/](https://frankkramer-lab.github.io/aucmedi/reference/)
- Getting Started: [https://frankkramer-lab.github.io/aucmedi/intro/](https://frankkramer-lab.github.io/aucmedi/intro/)
- Examples: [https://frankkramer-lab.github.io/aucmedi/examples/framework/](https://frankkramer-lab.github.io/aucmedi/examples/framework/)
- Tutorials: Coming soon.
- Applications: [https://frankkramer-lab.github.io/aucmedi/examples/applications/](https://frankkramer-lab.github.io/aucmedi/examples/applications/)
- PyPI package: [https://pypi.org/project/aucmedi/](https://pypi.org/project/aucmedi/)
- Docker Hub: Coming soon.

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
- [x] Extensive unittesting -> CI/CD
- [x] Available from PyPI for simple installation in various environments

**Planed milestones and features are:**
- [ ] Support for AutoML via CLI and Docker
- [ ] Examples for AutoML
- [ ] Documentation for AutoML
- [ ] Integration of bagging and stacking pipelines for utilizing ensemble learning techniques
- [ ] Publication

## Getting started: 60 seconds to automated medical image classification

**Install AUCMEDI via PyPI:**
```sh
pip install aucmedi
```

**Build a pipeline:**
```python
asd
```

**Train a model:**
```python
asd
```

**Make model predictions:**
```python
asd
```

## Lead Author

Dominik Müller\
Email: dominik.mueller@informatik.uni-augsburg.de\
IT-Infrastructure for Translational Medical Research\
University Augsburg\
Bavaria, Germany

## How to cite / More information

Coming soon.

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.\
See the LICENSE.md file for license rights and limitations.
