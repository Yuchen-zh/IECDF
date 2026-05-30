# -*-codeing = utf-8 -*-
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import os

def clipdata2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda(),
        'clip_image':batch[5].cuda(),
        'clip_text': batch[6].cuda()
    }
    return batch_data

def data2gpu(batch):
    batch_data = {
        'content': batch[0].cuda(),
        'content_masks': batch[1].cuda(),
        'label': batch[2].cuda(),
        'category': batch[3].cuda(),
        'image':batch[4].cuda()
    }
    return batch_data


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)




def _prepare_binary_prediction(y_pred, threshold=0.5):
    y_pred_arr = np.asarray(y_pred)

    if y_pred_arr.ndim > 1:
        if y_pred_arr.shape[1] == 1:
            y_pred_score = y_pred_arr[:, 0].astype(float)
            y_pred_label = (y_pred_score >= threshold).astype(int)
        else:
            y_pred_score = y_pred_arr[:, 1].astype(float)
            y_pred_label = np.argmax(y_pred_arr, axis=1).astype(int)

    else:
        if np.issubdtype(y_pred_arr.dtype, np.floating):
            y_pred_score = y_pred_arr.astype(float)
            y_pred_label = (y_pred_score >= threshold).astype(int)
        else:
            y_pred_label = y_pred_arr.astype(int)
            y_pred_score = y_pred_label.astype(float)

    return y_pred_score, y_pred_label


def _safe_auc(y_true, y_score):

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def _binary_counts_for_label(y_true, y_pred, positive_label):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    TP = np.sum((y_true == positive_label) & (y_pred == positive_label))
    FP = np.sum((y_true != positive_label) & (y_pred == positive_label))
    FN = np.sum((y_true == positive_label) & (y_pred != positive_label))
    TN = np.sum((y_true != positive_label) & (y_pred != positive_label))

    return TP, FP, FN, TN


def _metrics_from_counts(TP, FP, FN, TN):

    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    F1 = 2 * precision * recall / max(1e-8, precision + recall)

    specificity = TN / max(1, TN + FP)
    G_means = np.sqrt(recall * specificity)

    class_accuracy = TP / max(1, TP + FN)

    return {
        "Accuracy": class_accuracy,
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "specificity": specificity,
        "G_means": G_means,
    }


def _macro_binary_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
            "acc": 0.0,
        }

    TP_0, FP_0, FN_0, TN_0 = _binary_counts_for_label(y_true, y_pred, positive_label=0)
    TP_1, FP_1, FN_1, TN_1 = _binary_counts_for_label(y_true, y_pred, positive_label=1)

    m0 = _metrics_from_counts(TP_0, FP_0, FN_0, TN_0)
    m1 = _metrics_from_counts(TP_1, FP_1, FN_1, TN_1)

    return {
        "precision": (m0["precision"] + m1["precision"]) / 2,
        "recall": (m0["recall"] + m1["recall"]) / 2,
        "fscore": (m0["F1"] + m1["F1"]) / 2,
        "acc": accuracy_score(y_true, y_pred),
    }


def _format_metric_value(value):
    if value is None:
        return "nan"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def metrics(y_true, y_pred_score, y_pred_label, category, category_dict):

    y_true = np.asarray(y_true).astype(int)
    y_pred_score = np.asarray(y_pred_score).astype(float)
    y_pred_label = np.asarray(y_pred_label).astype(int)
    category = np.asarray(category)

    metrics_by_category = {}

    metrics_by_category["auc"] = _safe_auc(y_true, y_pred_score)
    metrics_by_category["metric"] = f1_score(
        y_true,
        y_pred_label,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    metrics_by_category["recall"] = recall_score(
        y_true,
        y_pred_label,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    metrics_by_category["precision"] = precision_score(
        y_true,
        y_pred_label,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    metrics_by_category["acc"] = accuracy_score(y_true, y_pred_label)

    reverse_category_dict = {v: k for k, v in category_dict.items()}

    res_by_category = {
        c_name: {
            "y_true": [],
            "y_pred_score": [],
            "y_pred_label": [],
        }
        for c_name in category_dict.keys()
    }

    for i, c in enumerate(category):
        c_name = reverse_category_dict.get(c, c)

        if c_name not in res_by_category:
            res_by_category[c_name] = {
                "y_true": [],
                "y_pred_score": [],
                "y_pred_label": [],
            }

        res_by_category[c_name]["y_true"].append(y_true[i])
        res_by_category[c_name]["y_pred_score"].append(y_pred_score[i])
        res_by_category[c_name]["y_pred_label"].append(y_pred_label[i])

    for c_name, res in res_by_category.items():
        y_true_c = np.asarray(res["y_true"]).astype(int)
        y_score_c = np.asarray(res["y_pred_score"]).astype(float)
        y_label_c = np.asarray(res["y_pred_label"]).astype(int)

        c_metrics = _macro_binary_metrics(y_true_c, y_label_c)
        c_metrics["auc"] = _safe_auc(y_true_c, y_score_c)

        metrics_by_category[c_name] = c_metrics

    return metrics_by_category


def metricsTrueFalse(
    y_true,
    y_pred,
    category,
    category_dict,
    output_file="metrics_results.txt",
    dataset_name="finefake",
    save_dir=None,
    domain_num=9,
):
    print(f"y_true shape: {np.asarray(y_true).shape}, dtype: {np.asarray(y_true).dtype}")
    print(f"y_pred shape: {np.asarray(y_pred).shape}, dtype: {np.asarray(y_pred).dtype}")

    y_true = np.asarray(y_true).astype(int)
    y_pred_score, y_pred_label = _prepare_binary_prediction(y_pred, threshold=0.5)

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower in ("weibo", "weibo21"):
        real_label = 0
        fake_label = 1
    else:
        real_label = 1
        fake_label = 0

    metrics_result = metrics(
        y_true=y_true,
        y_pred_score=y_pred_score,
        y_pred_label=y_pred_label,
        category=category,
        category_dict=category_dict,
    )

    TP_real, FP_real, FN_real, TN_real = _binary_counts_for_label(
        y_true,
        y_pred_label,
        positive_label=real_label,
    )
    real = _metrics_from_counts(TP_real, FP_real, FN_real, TN_real)

    TP_fake, FP_fake, FN_fake, TN_fake = _binary_counts_for_label(
        y_true,
        y_pred_label,
        positive_label=fake_label,
    )
    fake = _metrics_from_counts(TP_fake, FP_fake, FN_fake, TN_fake)

    metrics_result["real"] = real
    metrics_result["fake"] = fake

    overall_sensitivity = (real["recall"] + fake["recall"]) / 2
    overall_specificity = (real["specificity"] + fake["specificity"]) / 2
    metrics_result["G_means"] = np.sqrt(overall_sensitivity * overall_specificity)

    check_items = [
        ("F1", "metric", "F1"),
        ("Recall", "recall", "recall"),
        ("Precision", "precision", "precision"),
    ]

    for name, overall_key, class_key in check_items:
        avg = (real[class_key] + fake[class_key]) / 2
        overall = metrics_result[overall_key]

        if not np.isclose(overall, avg, rtol=1e-4, atol=1e-8):
            print(
                f"[Warning] Overall {name} ({overall:.4f}) "
                f"!= (real+fake)/2 ({avg:.4f})"
            )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, output_file)

        reserved_keys = {
            "acc",
            "precision",
            "recall",
            "metric",
            "auc",
            "G_means",
            "real",
            "fake",
        }

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("===== Overall Metrics =====\n")
            for key in ("acc", "precision", "recall", "metric", "auc", "G_means"):
                if key in metrics_result:
                    f.write(f"{key}: {_format_metric_value(metrics_result[key])}\n")

            f.write("\n===== Real News Metrics =====\n")
            for key in ("Accuracy", "precision", "recall", "F1", "specificity", "G_means"):
                if key in real:
                    f.write(f"{key}: {_format_metric_value(real[key])}\n")

            f.write("\n===== Fake News Metrics =====\n")
            for key in ("Accuracy", "precision", "recall", "F1", "specificity", "G_means"):
                if key in fake:
                    f.write(f"{key}: {_format_metric_value(fake[key])}\n")

            f.write("\n===== Per-Category Metrics =====\n")
            for c_name, c_metrics in sorted(metrics_result.items(), key=lambda x: str(x[0])):
                if c_name in reserved_keys:
                    continue

                if not isinstance(c_metrics, dict):
                    continue

                f.write(f"\n[{c_name}]\n")
                for key in ("precision", "recall", "fscore", "acc", "auc"):
                    if key in c_metrics:
                        f.write(f"  {key}: {_format_metric_value(c_metrics[key])}\n")

        print(f"Results saved to {save_path}")

    return metrics_result