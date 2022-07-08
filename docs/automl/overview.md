Wikipedia:  
AutoML potentially includes every stage from beginning with a raw dataset to building a machine learning model ready for deployment. AutoML was proposed as an artificial intelligence-based solution to the growing challenge of applying machine learning.[1][2] The high degree of automation in AutoML aims to allow non-experts to make use of machine learning models and techniques without requiring them to become experts in machine learning. Automating the process of applying machine learning end-to-end additionally offers the advantages of producing simpler solutions, faster creation of those solutions, and models that often outperform hand-designed models.


## AutoML Types in AUCMEDI

AUCMEDI integrates Automated Machine Learning (AutoML), which can be defined as a mentality to ensure easy application, integration and maintenance of complex medical image classification pipelines.

Therefore, AUCMEDI offers two interfaces for automatic building and fast application of state-of-the-art medical image classification pipelines.

!!! info "AutoML Overview of AUCMEDI"
    | AutoML Type | Required Software | Description |
    | ----------- | ----------------- | ----------- |
    | Command Line Interface (CLI) | Python & Dependencies | AUCMEDI application via command line script interaction in a local environment. Recommended for research settings. |
    | Docker Image         | Docker | AUCMEDI application via command line script interaction in a secure and isolated environment. Recommended for clinical settings. |

The CLI as well as Docker interface both utilize the same AutoML modes and pipeline principles.

![Figure: AUCMEDI AutoML](../images/aucmedi.automl.png)
*Flowchart diagram of AUCMEDI AutoML showing the pipeline workflow and three AutoML modes: training for model fitting, prediction for inference of unknown images, and evaluation for performance estimation.*
