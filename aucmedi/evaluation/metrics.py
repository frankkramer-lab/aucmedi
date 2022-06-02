import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def compute_metrics(preds, labels, n_labels, threshold=None):
    df_list = []
    for c in range(0, n_labels):
        # Initialize variables
        data_dict = {}

        # Identify truth and prediction for class c
        truth = labels[:, c]
        if threshold is None:
            pred_argmax = np.argmax(preds, axis=-1)
            pred = (pred_argmax == c).astype(np.int)
            pred_prob = np.max(preds, axis=-1)
        else:
            pred = np.where(preds[:, c] >= threshold, 1, 0)
            pred_prob = preds[:, c]

        # Compute the confusion matrix
        tp, tn, fp, fn = compute_CM(truth, pred)
        data_dict["TP"] = tp
        data_dict["TN"] = tn
        data_dict["FP"] = fp
        data_dict["FN"] = fn

        # Compute several metrics based on confusion matrix
        data_dict["Sensitivity"] = np.divide(tp, tp+fn)
        data_dict["Specificity"] = np.divide(tn, tn+fp)
        data_dict["Precision"] = np.divide(tp, tp+fp)
        data_dict["FPR"] = np.divide(fp, fp+tn)
        data_dict["FNR"] = np.divide(fn, fn+tp)
        data_dict["FDR"] = np.divide(fp, fp+tp)
        data_dict["Accuracy"] = np.divide(tp+tn, tp+tn+fp+fn)
        data_dict["F1"] = np.divide(2*tp, 2*tp+fp+fn)

        # Compute area under the ROC curve
        data_dict["AUC"] = roc_auc_score(truth, pred_prob)

        # Parse metrics to dataframe
        df = pd.DataFrame.from_dict(data_dict, orient="index",
                                    columns=["score"])
        df = df.reset_index()
        df.rename(columns={"index": "metric"}, inplace=True)
        df["class"] = c

        # Append dataframe to list
        df_list.append(df)

    # Combine dataframes
    df_final = pd.concat(df_list, axis=0, ignore_index=True)
    # Return final dataframe
    return df_final

def compute_confusion_matrix(preds, labels, n_labels):
    preds_argmax = np.argmax(preds, axis=-1)
    labels_argmax = np.argmax(labels, axis=-1)
    rawcm = np.zeros((n_labels, n_labels))
    for i in range(0, labels.shape[0]):
        rawcm[labels_argmax[i]][preds_argmax[i]] += 1
    return rawcm


def compute_roc(preds, labels, n_labels):
    fpr_list = []
    tpr_list = []
    for i in range(0, n_labels):
        truth_class = labels[:, i].astype(int)
        pdprob_class = preds[:, i]
        fpr, tpr, _ = roc_curve(truth_class, pdprob_class)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return fpr_list, tpr_list



# Compute confusion matrix
def compute_CM(gt, pd):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, len(gt)):
        if gt[i] == 1 and pd[i] == 1 : tp += 1
        elif gt[i] == 1 and pd[i] == 0 : fn += 1
        elif gt[i] == 0 and pd[i] == 0 : tn += 1
        elif gt[i] == 0 and pd[i] == 1 : fp += 1
        else : print("ERROR at confusion matrix", i)
    return tp, tn, fp, fn
