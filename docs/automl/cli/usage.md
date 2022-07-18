This chapter demonstrates the most important processing steps of an AUCMEDI AutoML CLI usage.

The dataset structure is by default in the working directory for CLI.

## Training

For the start, our working directory must contain the subdirectory `training`.
The `training` directory must contain all images for model training, sorted class subdirectories.

> show file structure of current working directory

```bash title="$ ls -l . "
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
```

> show class structure of training directory

```bash title="$ ls -l training/"
drwxr-xr-x 2 root root 57344 Jul 17 23:02 01_TUMOR
drwxr-xr-x 2 root root 65536 Jul 17 23:02 02_STROMA
drwxr-xr-x 2 root root 49152 Jul 17 23:02 03_COMPLEX
drwxr-xr-x 2 root root 61440 Jul 17 23:02 04_LYMPHO
drwxr-xr-x 2 root root 65536 Jul 17 23:02 05_DEBRIS
drwxr-xr-x 2 root root 57344 Jul 17 23:02 06_MUCOSA
drwxr-xr-x 2 root root 57344 Jul 17 23:02 07_ADIPOSE
drwxr-xr-x 2 root root 53248 Jul 17 23:02 08_EMPTY
```

> show file structure of a class directory

```bash title="$ ls -l training/01_TUMOR/"
-rw-r--r-- 1 root root 68376 Jul 17 23:02 10009_CRC-Prim-HE-03_009.tif_Row_301_Col_151.tif
-rw-r--r-- 1 root root 68376 Jul 17 23:02 10062_CRC-Prim-HE-02_003b.tif_Row_1_Col_301.tif
-rw-r--r-- 1 root root 68374 Jul 17 23:02 100B0_CRC-Prim-HE-09_009.tif_Row_1_Col_301.tif
-rw-r--r-- 1 root root 68376 Jul 17 23:02 10104_CRC-Prim-HE-10_021.tif_Row_451_Col_1.tif
-rw-r--r-- 1 root root 68376 Jul 17 23:02 10142_CRC-Prim-HE-09_025.tif_Row_151_Col_151.tif
...
```

In order to create a high-performance model for clinical decision support, it is
required to have one or multiple already fitted models for this imaging task.

In our usage example, we will train a ResNet50 model from scratch.

> run AUCMEDI AutoML training hub

```bash title="$ aucmedi training --architecture ResNet50 --epochs 25"
2022-07-18 12:57:25.282772: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/10
2022-07-18 12:57:32.516662: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8100
177/177 [==============================] - ETA: 0s - loss: 0.5103 - auc: 0.9563 - f1_score: 0.7556   
Epoch 1: val_loss improved from inf to 0.20766, saving model to model/model.best_loss.hdf5
177/177 [==============================] - 21s 92ms/step - loss: 0.5103 - auc: 0.9563 - f1_score: 0.7556 - val_loss: 0.2077 - val_auc: 0.9864 - val_f1_score: 0.8958 - lr: 1.0000e-04
Epoch 2/10
177/177 [==============================] - ETA: 0s - loss: 0.1932 - auc: 0.9893 - f1_score: 0.8842
Epoch 2: val_loss improved from 0.20766 to 0.18348, saving model to model/model.best_loss.hdf5
177/177 [==============================] - 15s 84ms/step - loss: 0.1932 - auc: 0.9893 - f1_score: 0.8842 - val_loss: 0.1835 - val_auc: 0.9891 - val_f1_score: 0.9010 - lr: 1.0000e-04
...
Epoch 23/25
177/177 [==============================] - ETA: 0s - loss: 0.0087 - auc: 0.9999 - f1_score: 0.9894
Epoch 23: val_loss did not improve from 0.12966
Epoch 23: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-07.
177/177 [==============================] - 22s 126ms/step - loss: 0.0087 - auc: 0.9999 - f1_score: 0.9894 - val_loss: 0.1477 - val_auc: 0.9933 - val_f1_score: 0.9374 - lr: 1.0000e-05
Epoch 24/25
177/177 [==============================] - ETA: 0s - loss: 0.0051 - auc: 1.0000 - f1_score: 0.9950
Epoch 24: val_loss did not improve from 0.12966
177/177 [==============================] - 22s 125ms/step - loss: 0.0051 - auc: 1.0000 - f1_score: 0.9950 - val_loss: 0.1377 - val_auc: 0.9934 - val_f1_score: 0.9413 - lr: 1.0000e-06
Epoch 25/25
177/177 [==============================] - ETA: 0s - loss: 0.0061 - auc: 1.0000 - f1_score: 0.9941
Epoch 25: val_loss did not improve from 0.12966
177/177 [==============================] - 22s 125ms/step - loss: 0.0061 - auc: 1.0000 - f1_score: 0.9941 - val_loss: 0.1374 - val_auc: 0.9941 - val_f1_score: 0.9400 - lr: 1.0000e-06
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 6.4 x 4.8 in image.
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: model/plot.fitting_course.png
```

The result of the AUCMEDI AutoML training hub is a `model` directory in the working directory.

It contains one or multiple AUCMEDI models with other metadata created during the fitting process.

> show file structure of current working directory (after training)

```bash title="$ ls -l ."
drwxr-xr-x  2 root root 4096 Jul 18 13:05 model
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
```

> show model directory

```bash title="$ ls -l model/"
-rw-r--r-- 1 root root      3028 Jul 18 13:05 logs.training.csv
-rw-r--r-- 1 root root       444 Jul 18 12:57 meta.training.json
-rw-r--r-- 1 root root 107530168 Jul 18 12:58 model.best_loss.hdf5
-rw-r--r-- 1 root root 296305016 Jul 18 13:05 model.last.hdf5
-rw-r--r-- 1 root root     80927 Jul 18 13:05 plot.fitting_course.png
```

More information about the parameters for training can be found here:
[AutoML - Parameters - training](../../parameters/#automl-mode-training).

## Inference

For predicting the classification of unknown images, the images should be stored
in the `test` directory.


> show file structure of current working directory (after training)

```bash title="$ ls -l ."
drwxr-xr-x  2 root root 4096 Jul 18 13:05 model
drwxr-xr-x  2 root root 4096 Jul 18 13:10 test
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
```

> show file structure of test directory

```bash title="$ ls -l test/"
-rw-r--r-- 1 root root 68376 Jun 28 15:43 UNKNOWN_IMAGE.0001.tif
-rw-r--r-- 1 root root 68376 Jun 28 15:43 UNKNOWN_IMAGE.0002.tif
-rw-r--r-- 1 root root 68364 Jun 28 15:43 UNKNOWN_IMAGE.0003.tif
-rw-r--r-- 1 root root 68378 Jun 28 15:43 UNKNOWN_IMAGE.0004.tif
```

The AUCMEDI AutoML prediction hub read out the pipeline configuration and fitted models
from the provided `model` directory.

> run AUCMEDI AutoML prediction hub

```bash title="$ aucmedi prediction"
2022-07-18 13:13:47.334439: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-07-18 13:13:52.783231: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8100
4/4 [==============================] - 3s 87ms/step
```

The results will be stored in a CSV file called `preds.csv` (by default).

The CSV file shows the classification probability for an image for each class.

> show file structure of current working directory (after prediction)

```bash title="$ ls -l ."
drwxr-xr-x  2 root root 4096 Jul 18 13:05 model
-rw-r--r--  1 root root  566 Jul 18 13:13 preds.csv
drwxr-xr-x  2 root root 4096 Jul 18 13:10 test
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
```

> show inference content of prediction file

```bash title="$ cat preds.csv"
SAMPLE,01_TUMOR,02_STROMA,03_COMPLEX,04_LYMPHO,05_DEBRIS,06_MUCOSA,07_ADIPOSE,08_EMPTY
UNKNOWN_IMAGE.0001,0.9947149,5.093858e-05,0.0032877475,0.0004719145,0.00061258266,0.0005081127,1.5236534e-05,0.0003385503
UNKNOWN_IMAGE.0002,0.12757735,0.3084325,0.52998906,0.008813165,0.012200621,0.01229311,0.00034778274,0.00034644845
UNKNOWN_IMAGE.0003,0.9978336,4.6700584e-06,0.000806501,6.4442225e-05,0.0011141102,6.125228e-05,5.657194e-05,5.8843718e-05
UNKNOWN_IMAGE.0004,9.7639786e-05,0.0030071975,0.7069594,0.27908832,0.0037088492,0.0069722794,3.9823564e-05,0.00012642879
```

More information about the parameters for training can be found here:
[AutoML - Parameters - prediction](../../parameters/#automl-mode-prediction).

## Evaluation

For performance estimation of the model, a `validation` set is required which means
the classification prediction of images with a known class annotation.

In order to demonstrate the CSV annotation, as well, the validation data is encoded
in the following file structure:

> show file structure of current working directory (for evaluation)

```bash title="$ ls -l ."
drwxr-xr-x  2 root root 4096 Jul 18 13:05 model
-rw-r--r--  1 root root  566 Jul 18 13:13 preds.csv
drwxr-xr-x  2 root root 4096 Jul 18 13:10 test
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
drwxr-xr-x  3 root root 4096 Jul 18 13:20 validation
```

> show file structure of validation set

```bash title="$ ls -l validation/"
-rw-r--r-- 1 root root 1419 Jul 18 13:20 annotations.csv
drwxr-xr-x 2 root root 4096 Jul 18 13:20 images
```

> show file structure of the validation image directory

```bash title="$ ls -l validation/images/"
-rw-r--r-- 1 root root 67926 Jul 17 23:02 10070_CRC-Prim-HE-04_036.tif_Row_601_Col_601.tif
-rw-r--r-- 1 root root 68374 Jul 17 23:02 10078_CRC-Prim-HE-03_001.tif_Row_151_Col_601.tif
-rw-r--r-- 1 root root 68376 Jul 17 23:02 1012B_CRC-Prim-HE-10_016.tif_Row_1_Col_301.tif
...
```

> show annotation content of validation CSV file

```bash title="$ cat validation/annotations.csv"
SAMPLE,CLASS
101A0_CRC-Prim-HE-03_034.tif_Row_151_Col_1.tif,01_TUMOR
1012B_CRC-Prim-HE-10_016.tif_Row_1_Col_301.tif,05_DEBRIS
13111_CRC-Prim-HE-05_009a.tif_Row_751_Col_1351.tif,03_COMPLEX
...
```

The evaluation mode of AUCMEDI requires another prediction call for the new
validation images.

> Compute predictions for validation images with a specific input & output path

```bash title="$ aucmedi prediction --path_imagedir validation/images/ --path_pred validation/preds.csv"
2022-07-18 13:26:25.699603: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-07-18 13:26:31.281593: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8100
20/20 [==============================] - 4s 50ms/step
```

> show inference content of prediction file

```bash title="$ cat validation/preds.csv"
SAMPLE,01_TUMOR,02_STROMA,03_COMPLEX,04_LYMPHO,05_DEBRIS,06_MUCOSA,07_ADIPOSE,08_EMPTY
10070_CRC-Prim-HE-04_036.tif_Row_601_Col_601,0.0002848482,0.031636566,0.00048324154,0.00043562692,0.042952724,0.00074527983,0.92224264,0.001219118
10078_CRC-Prim-HE-03_001.tif_Row_151_Col_601,0.0065458445,0.486334,0.20805378,8.415832e-05,0.27862436,0.019759992,4.8646994e-05,0.0005491826
1012B_CRC-Prim-HE-10_016.tif_Row_1_Col_301,0.0030586768,0.13436078,0.030891698,0.004924605,0.7850058,0.021087538,0.015678901,0.00499198
...
```

Afterwards, it is possible to estimate the performance based on the annotations
and predicted classifications of the validation set.

> compute performance via AUCMEDI AutoML evaluation

```bash title="$ aucmedi evaluation --path_imagedir validation/images/ --path_gt validation/annotations.csv --path_pred validation/preds.csv"
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 12 x 9 in image.
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: evaluation/plot.performance.barplot.png
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 10 x 9 in image.
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: evaluation/plot.performance.confusion_matrix.png
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:719: PlotnineWarning: Saving 10 x 9 in image.
/usr/local/lib/python3.8/dist-packages/plotnine/ggplot.py:722: PlotnineWarning: Filename: evaluation/plot.performance.roc.png
```

> show file structure of current working directory (after evaluation)

```bash title="$ ls -l ."
drwxr-xr-x  2 root root 4096 Jul 18 13:31 evaluation
drwxr-xr-x  2 root root 4096 Jul 18 13:05 model
-rw-r--r--  1 root root  566 Jul 18 13:13 preds.csv
drwxr-xr-x  2 root root 4096 Jul 18 13:10 test
drwxr-xr-x 10 root root 4096 Jul 17 23:02 training
drwxr-xr-x  3 root root 4096 Jul 18 13:20 validation
```

> show file structure of evaluation directory

```bash title="$ ls -l evaluation/"
-rw-r--r-- 1 root root   2537 Jul 18 13:31 metrics.performance.csv
-rw-r--r-- 1 root root 191015 Jul 18 13:31 plot.performance.barplot.png
-rw-r--r-- 1 root root 150348 Jul 18 13:31 plot.performance.confusion_matrix.png
-rw-r--r-- 1 root root 138702 Jul 18 13:31 plot.performance.roc.png
```

![Figure: Results](../../images/aucmedi.automl.usage.plot.png)
*Resulting evaluation result of AUCMEDI AutoML CLI usage example.
File: evaluation/plot.performance.confusion_matrix.png.*

More information about the parameters for training can be found here:
[AutoML - Parameters - evaluation](../../parameters/#automl-mode-evaluation).
