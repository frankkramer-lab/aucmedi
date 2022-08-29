The AutoML examples for AUCMEDI are demonstrated in Markdown.

## Overview

| Data Set | Task | AutoML Type | Link  |
|-------------|:--------------------:|:------:|:------:|
| [CHMNIST](#CHMNIST) | Multi-Class classifier for Pathology | CLI | [AutoML - CLI - Usage](../../automl/cli/usage/) |
| [CHMNIST](#CHMNIST) | Multi-Class classifier for Pathology | Docker | [AutoML - Docker - Usage](../../automl/docker/usage/) |


## <a name="CHMNIST"></a>Colorectal Histology MNIST

**Classes:** 8 - EMPTY, COMPLEX, MUCOSA, DEBRIS, ADIPOSE, STROMA, LYMPHO, TUMOR  
**Size:** 5.000 images  
**Source:** https://www.kaggle.com/kmader/colorectal-histology-mnist  

**Short Description:**  
Automatic recognition of different tissue types in histological images is an essential part in the digital pathology toolbox. Texture analysis is commonly used to address this problem; mainly in the context of estimating the tumour/stroma ratio on histological samples. However, although histological images typically contain more than two tissue types, only few studies have addressed the multi-class problem. For colorectal cancer, one of the most prevalent tumour types, there are in fact no published results on multiclass texture separation. The dataset serves as a much more interesting MNIST or CIFAR10 problem for biologists by focusing on histology tiles from patients with colorectal cancer. In particular, the data has 8 different classes of tissue (but Cancer/Not Cancer can also be an interesting problem).

**Reference:**  
Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zöllner FG. Multi-class texture analysis in colorectal cancer histology. Sci Rep. 2016 Jun 16;6:27988. doi: 10.1038/srep27988. PMID: 27306927; PMCID: PMC4910082.
