## Dataset Setup

The AUCMEDI AutoML expects a fixed dataset structure with default parameters.

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

This dataset structure can be customized with the following parameters:

table

automl mode, input or output, para, default value
