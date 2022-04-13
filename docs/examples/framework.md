The framework/API examples for AUCMEDI are implemented in Jupyter Notebooks.   
Jupyter Notebooks offer reproducibility by including the output of each coding block, but can also integrate commentary blocks with Markdown. Also, Jupyter Notebooks can be directly displayed in GitHub without any additional software.

## Overview

| Data Set | Task | Link  |
|-------------|:--------------------:|:------:|
| [Pneumonia](#Pneumonia) | Ensemble Learning: Bagging via 3-fold Cross-Validation | [xray_pneumonia.ipynb](https://github.com/frankkramer-lab/aucmedi/blob/master/examples/framework/xray_pneumonia.ipynb) |


## <a name="Pneumonia"></a>Viral Pneumonia Detection via Ensemble Learning enhanced Image Classifier

In this work we use the AUCMEDI-Framework to train a deep neural network to classify chest X-ray images as either normal or viral pneumonia. Stratified k-fold cross-validation with k=3 is used to generate the validation-set and 15% of the data are set aside for the evaluation of the models of the different folds and ensembles each. A random-forest ensemble as well as a Soft-Majority-Vote ensemble are built from the predictions of the different folds. Evaluation metrics (Classification-Report, macro f1-scores, Confusion-Matrices, ROC-Curves) of the individual folds and the ensembles show that the classifier works well. Finally Grad-CAM and LIME explainable artificial intelligence (XAI) algorithms are applied to visualize the image features that are most important for the prediction. For Grad-CAM the heatmaps of the three folds are furthermore averaged for all images in order to calculate a mean XAI-heatmap. As the heatmaps of the different folds for most images differ only slightly this averaging procedure works well. However, only medical professionals can evaluate the quality of the features marked by the XAI. A comparison of the evaluation metrics with metrics of standard procedures such as PCR would also be important. Further limitations are discussed.

A manuscript describing this application was uploaded to arXiv:  
[https://arxiv.org/abs/2110.01017](https://arxiv.org/abs/2110.01017)
