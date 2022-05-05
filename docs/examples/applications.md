AUCMEDI was already / is currently applied in multiple projects, challenges or clinical studies.  
The Code for these applications is presented in separate GitHub repositories.

Even so these implementations are more complex than standard examples, they represent state-of-the-art and functional pipelines which were top-ranked in challenges or are successfully integrated in clinical environments.

## Overview

| Application | Type | Link  |
|-------------|:--------------------:|:------:|
| [Multi-Disease Detection in Retinal Imaging](#RIADD) | Challenge (rank #7) | [RIADD - Grand Challenge](https://github.com/frankkramer-lab/riadd.aucmedi) |
| [Ensemble Learning Analysis](#ENSMIC) | Research | [ENSMIC](https://github.com/frankkramer-lab/ensmic) |
| COVID-19 Severity Detection | Challenge (rank #4 - ongoing) | [COVID-19 Algorithm - Grand Challenge](https://grand-challenge.org/algorithms/stoic2021-covid-19-lung-ct-scans-team-augsburg) |
| Pathology - Gleason Score Estimation | Clinical Study | Ongoing |


## <a name="RIADD"></a>Multi-Disease Detection in Retinal Imaging

Preventable or undiagnosed visual impairment and blindness affects billion of people worldwide. Automated multi-disease detection models offer great potential to address this problem via clinical decision support in diagnosis. In this work, we proposed an innovative multi-disease detection pipeline for retinal imaging which utilizes ensemble learning to combine the predictive power of several heterogeneous deep convolutional neural network models. Our pipeline includes state-of-the-art strategies like transfer learning, class weighting, real-time image augmentation and focal loss utilization. Furthermore, we integrated ensemble learning techniques like heterogeneous deep learning models, bagging via 5-fold cross-validation and stacked logistic regression models.

Participation at the Retinal Image Analysis for multi-Disease Detection Challenge (RIADD):
[https://riadd.grand-challenge.org/](https://riadd.grand-challenge.org/)

**Reference:**  
Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2021)  
Multi-Disease Detection in Retinal Imaging Based on Ensembling Heterogeneous Deep Learning Models  
[https://pubmed.ncbi.nlm.nih.gov/34545816/](https://pubmed.ncbi.nlm.nih.gov/34545816/)

## <a name="ENSMIC"></a>Ensemble Learning Analysis

Novel and high-performance medical image classification pipelines are heavily utilizing ensemble learning strategies. The idea of ensemble learning is to assemble diverse models or multiple predictions and, thus, boost prediction performance. However, it is still an open question to what extend as well as which ensemble learning strategies are beneficial in deep learning based medical image classification pipelines.

In this work, we proposed a reproducible medical image classification pipeline (ensmic) for analyzing the performance impact of the following ensemble learning techniques: Augmenting, Stacking, and Bagging. The pipeline consists of state-of-the-art preprocessing and image augmentation methods as well as nine deep convolution neural network architectures. It was applied on four popular medical imaging datasets with varying complexity. Furthermore, 12 pooling functions for combining multiple predictions were analyzed, ranging from simple statistical functions like unweighted averaging up to more complex learning-based functions like support vector machines.

Our results revealed that Stacking achieved the largest performance gain of up to 13% F1-score increase. Augmenting showed consistent improvement capabilities by up to 4% and is also applicable to single model based pipelines. Cross-validation based Bagging demonstrated significant performance gain close to Stacking, which resulted in an F1-score increase up to +11%. Furthermore, we demonstrated that simple statistical pooling functions are equal or often even better than more complex pooling functions. We concluded that the integration of ensemble learning techniques is a powerful method for any medical image classification pipeline to improve robustness and boost performance.

**Reference:**  
Dominik Müller, Iñaki Soto-Rey and Frank Kramer. (2022)  
An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks.  
arXiv e-print: [https://arxiv.org/abs/2201.11440](https://arxiv.org/abs/2201.11440)
