## AutoML Types in AUCMEDI

AUCMEDI offers a CLI and Docker interface for automatic building and fast application of state-of-the-art medical image classification pipelines.

!!! info "AutoML Overview of AUCMEDI"
    ![Figure: AUCMEDI AutoML](../images/aucmedi.automl.png)
    *Flowchart diagram of AUCMEDI AutoML showing the pipeline workflow and three AutoML modes: training for model fitting, prediction for inference of unknown images, and evaluation for performance estimation.*

## Dataset Setup

AUCMEDI AutoML expects a fixed dataset structure if run on default parameters.  
The dataset structure is by default in the working directory for CLI or
is mounted as volume into the container for Docker.

```bash
aucmedi.data/
├── training/                     # Required for training
│   ├── class_a/
│   │   ├── img_x.png
│   │   └── ...
│   ├── class_b/                  # Subdirectory for each class
│   │   ├── img_y.png
│   │   └── ...
│   ├── class_c/
│   │   ├── img_z.png             # Image names have to be unique
│   │   └── ...                   # between subdirectories
│   └── ...
├── test/                         # Required for prediction
│   ├── unknown_img_n.png
│   └── ...
├── model/                        # Will be created by training
├── evaluation/                   # Will be created by evaluation
└── preds.csv                     # Will be created by prediction
```

## Basic Usage - CLI

This example demonstrates the basic installation and application of AUCMEDI AutoML with the CLI.
The dataset have to be located in the working directory (inside of `aucmedi.data/`).

**Install AUCMEDI via PyPI**
```sh
pip install aucmedi
```

**Train a model and classify unknown images**
```bash
# Run training with default arguments, but a specific architecture
aucmedi training --architecture "DenseNet121"

# Run prediction with default arguments
aucmedi prediction
```

## Basic Usage - Docker

This example demonstrates the basic installation and application of AUCMEDI AutoML with Docker.
The dataset have to be mounted with a volume (with an absolute file path like in the example).

**Install AUCMEDI via GitHub Container Registry**
```sh
docker pull ghcr.io/frankkramer-lab/aucmedi:latest
```

**Train a model and classify unknown images**
```bash
# Run training with default arguments, but a specific architecture
docker run \
  -v /home/dominik/aucmedi.data:/data \
  --rm \
  ghcr.io/frankkramer-lab/aucmedi:latest \
  training \
  --architecture "DenseNet121"

# Run prediction with default arguments
docker run \
  -v /home/dominik/aucmedi.data:/data \
  --rm \
  ghcr.io/frankkramer-lab/aucmedi:latest \
  prediction
```

## More Details

More examples can be found here:
[Examples - AutoML](../../examples/automl/)

The full documentation for AUCMEDI AutoML can be found here:
[AutoML - Overview](../../automl/overview/)
