![aucmedi_logo](../images/aucmedi.logo.description.png)

The field of AI-based medical image classification has experienced rapid growth in recent years. The reason for this can be traced back to the increasing availability of medical imaging as well as the improvement in the predictive capabilities of automated image analysis through deep neural networks⁠. The possible integration of such deep neural networks into clinical routine is therefore currently a highly relevant research topic. Hence, clinicians, especially the imaging disciplines, are using models as clinical decision support in order to improve diagnostic certainty or to automate time-consuming manual processes.

However, clinical application studies reveal that the integration of image classification pipelines into a hospital environment presents significant challenges. Solutions from the literature that have already been implemented are usually independent software, so-called isolated solutions, which have been developed and optimized for a specific disease or a single data set⁠. Due to the lack of generalizability, clinicians are faced with the problem of reusability in their own data sets and therefore no practical use in clinical research is possible.

The open-source Python framework AUCMEDI offers a solution for the challenges described. The software package not only offers a **library as a high-level API** for the standardized construction of modern medical image classification pipelines, but also reproducible installation and **direct application via CLI or Docker**. With AUCMEDI, researchers are able to set up a complete and easy-to-integrate pipeline for medical image classification with just a few lines of code.

AUCMEDI provides two modules:  
- The framework or API for building medical image classification pipelines in Python  
- The AutoML module via CLI or Docker for fast application  

## Philosophy of AUCMEDI

**User friendliness:**  
AUCMEDI is an intuitive API designed for human beings, not machines. Building state-of-the-art medical image classification pipelines does not have to mean re-inventing the wheel for every user. To meet the continuously growing interest in medical imaging processing, AUCMEDI provides consistent and simple APIs for minimizing the number of user actions required for common use cases.

**Modularity:**  
The general steps in medical image processing are identical for nearly all projects. Nevertheless, switching to another neural network architecture or data set format breaks most of the publicly available medical image processing software today. AUCMEDI changes that! In particular, data I/O, pre-/postprocessing functions, metrics and model architectures are standalone interfaces that are easily interchangeable.

**Easy extensibility:**  
Contributions are simple to integrate into the AUCMEDI pipeline. AUCMEDI provides interfaces and abstract base classes for all types of classes or functions. That facilitates defining the structure and setup of 3rd party code and results in easy integration of e.g. architectures, subfunctions or adapting AUCMEDI to your data structure.

## Support

For help on how to use AUCMEDI, check out the tutorials, examples and further documentation.

If you have any problem or identified a bug, please use the issue system of our [GitHub repository](https://github.com/frankkramer-lab/aucmedi).  
Share your problems and contribute to improve AUCMEDI.

## Why this name, AUCMEDI?

Maybe, you are asking yourself: What abbreviation AUCMEDI stands for?  
The answer is quite simple: <b>Au</b>tomated <b>C</b>lassification of <b>Med</b>ical <b>I</b>mages.

And how should I pronounce it correctly?  
Answer: AUC-MEDI

As two words:  
- AUC as in AUROC (area under the curve of a receiver operating characteristic)  
- MEDI as in MEDICAL  
