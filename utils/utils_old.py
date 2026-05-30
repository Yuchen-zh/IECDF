# -*-codeing = utf-8 -*-
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

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

def metricsTrueFalse(y_true, y_pred, category, category_dict, output_file="metrics_results.txt", dataset_name='finefake', save_dir=None, domain_num=9):
    print(f"y_true shape: {np.array(y_true).shape}, dtype: {np.array(y_true).dtype}")
    print(f"y_pred shape: {np.array(y_pred).shape}, dtype: {np.array(y_pred).dtype}")

    if np.issubdtype(np.array(y_pred).dtype, np.floating):
        y_pred = (np.array(y_pred) > 0.5).astype(int)

    if len(np.array(y_pred).shape) > 1 and np.array(y_pred).shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    y_GT = y_true
    metricsTrueFalse = metrics(y_true, y_pred, category, category_dict)

    
    fake = {}
    real = {}

    THRESH = [0.5]
    realnews_TP, realnews_TN, realnews_FP, realnews_FN = [0] * domain_num, [0] * domain_num, [0] * domain_num, [0] * domain_num
    fakenews_TP, fakenews_TN, fakenews_FP, fakenews_FN = [0] * domain_num, [0] * domain_num, [0] * domain_num, [0] * domain_num
    realnews_sum, fakenews_sum = [0] * domain_num, [0] * domain_num
    for thresh_idx, thresh in enumerate(THRESH):
        for i in range(len(y_pred)):
            if y_pred[i]< thresh:y_pred[i]=0
            else:y_pred[i]=1
        for idx in range(len(y_pred)):
            current_category = category[idx]
            is_fake = (y_GT[idx] == 1) if dataset_name in ('weibo', 'weibo21') else (y_GT[idx] == 0)
            if is_fake:
                #  FAKE NEWS RESULT
                fakenews_sum[thresh_idx] += 1
                if y_pred[idx] == 0:
                    fakenews_FN[thresh_idx] += 1
                    realnews_FP[thresh_idx] += 1
                else:
                    fakenews_TP[thresh_idx] += 1
                    realnews_TN[thresh_idx] += 1
            else:
                # REAL NEWS RESULT
                realnews_sum[thresh_idx] += 1
                if y_pred[idx] == 1:
                    realnews_FN[thresh_idx] += 1
                    fakenews_FP[thresh_idx] += 1
                else:
                    realnews_TP[thresh_idx] += 1
                    fakenews_TN[thresh_idx] += 1

    val_accuracy, real_accuracy, fake_accuracy = [0] * domain_num, [0] * domain_num, [0] * domain_num
    real_precision, fake_precision, real_recall, fake_recall = [0] * domain_num, [0] * domain_num, [0] * domain_num, [0] * domain_num
    real_F1, fake_F1 = [0] * domain_num, [0] * domain_num

    for thresh_idx, _ in enumerate(THRESH):
        val_accuracy[thresh_idx] = (realnews_TP[thresh_idx] + realnews_TN[thresh_idx]) / max(1, realnews_TP[thresh_idx] + realnews_TN[thresh_idx] + realnews_FP[thresh_idx] + realnews_FN[thresh_idx])
        real_accuracy[thresh_idx] = realnews_TP[thresh_idx] / max(1, realnews_sum[thresh_idx])
        fake_accuracy[thresh_idx] = fakenews_TP[thresh_idx] / max(1, fakenews_sum[thresh_idx])

        real_precision[thresh_idx] = realnews_TP[thresh_idx] / max(1, realnews_TP[thresh_idx] + realnews_FP[thresh_idx])
        fake_precision[thresh_idx] = fakenews_TP[thresh_idx] / max(1, fakenews_TP[thresh_idx] + fakenews_FP[thresh_idx])

        real_recall[thresh_idx] = realnews_TP[thresh_idx] / max(1, realnews_TP[thresh_idx] + realnews_FN[thresh_idx])
        fake_recall[thresh_idx] = fakenews_TP[thresh_idx] / max(1, fakenews_TP[thresh_idx] + fakenews_FN[thresh_idx])

        real_F1[thresh_idx] = 2 * (real_recall[thresh_idx] * real_precision[thresh_idx]) / max(1, real_recall[thresh_idx] + real_precision[thresh_idx])
        fake_F1[thresh_idx] = 2 * (fake_recall[thresh_idx] * fake_precision[thresh_idx]) / max(1, fake_recall[thresh_idx] + fake_precision[thresh_idx])

    
    total_TP = realnews_TP[0] + fakenews_TP[0]
    total_FP = realnews_FP[0] + fakenews_FP[0]

    fake['precision'] = fake_precision[0]
    fake['recall'] = fake_recall[0]
    fake['F1'] = fake_F1[0]
    fake["Accuracy"] = fake_accuracy[0]

    real['precision'] = real_precision[0]
    real['recall'] = real_recall[0]
    real['F1'] = real_F1[0]
    real["Accuracy"] = real_accuracy[0]

    metricsTrueFalse['real'] = real
    metricsTrueFalse['fake'] = fake

    # 验证：总体 macro 指标应等于 (real + fake) / 2
    for name, overall_key, real_key, fake_key in [
        ('F1',       'metric',    'F1',         'F1'),
        ('Recall',   'recall',    'recall',     'recall'),
        ('Precision','precision', 'precision',  'precision'),
    ]:
        avg = (real[real_key] + fake[fake_key]) / 2
        overall = metricsTrueFalse[overall_key]
        if not np.isclose(overall, avg, rtol=1e-4):
            print(f"[Warning] Overall {name} ({overall:.4f}) != (real+fake)/2 ({avg:.4f})")

    # 保存结果到文件
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, output_file)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("===== Overall Metrics =====\n")
            for key in ('acc', 'precision', 'recall', 'metric', 'auc'):
                if key in metricsTrueFalse:
                    f.write(f"{key}: {metricsTrueFalse[key]:.4f}\n")
            f.write("\n===== Real News Metrics =====\n")
            for key in ('Accuracy', 'precision', 'recall', 'F1'):
                if key in real:
                    f.write(f"{key}: {real[key]:.4f}\n")
            f.write("\n===== Fake News Metrics =====\n")
            for key in ('Accuracy', 'precision', 'recall', 'F1'):
                if key in fake:
                    f.write(f"{key}: {fake[key]:.4f}\n")
            f.write("\n===== Per-Category Metrics =====\n")
            for c_name, c_metrics in sorted(metricsTrueFalse.items()):
                if isinstance(c_metrics, dict) and 'precision' in c_metrics:
                    f.write(f"\n[{c_name}]\n")
                    for key in ('precision', 'recall', 'fscore', 'acc', 'auc'):
                        if key in c_metrics:
                            f.write(f"  {key}: {c_metrics[key]:.4f}\n")
        print(f"Results saved to {save_path}")

    return metricsTrueFalse


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    # 每个领域 AUC — 用原始概率算，不需要二值化
    for c, res in res_by_category.items():
        metrics_by_category[c] = {
                'auc': round(roc_auc_score(res['y_true'], res['y_pred']), 4)
            }

    # 总体指标（sklearn）
    metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='micro')
    y_pred_bin = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred_bin, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred_bin, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred_bin, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred_bin)

    # 每个领域 precision / recall / F1 / acc — 手动公式计算
    for c, res in res_by_category.items():
        y_true_c = np.array(res['y_true'])
        y_pred_c = np.around(np.array(res['y_pred'])).astype(int)

        TP = np.sum((y_true_c == 1) & (y_pred_c == 1))
        TN = np.sum((y_true_c == 0) & (y_pred_c == 0))
        FP = np.sum((y_true_c == 0) & (y_pred_c == 1))
        FN = np.sum((y_true_c == 1) & (y_pred_c == 0))

        # class 1: weibo→fake, finefake→real
        p1 = TP / max(1, TP + FP)
        r1 = TP / max(1, TP + FN)
        f1_1 = 2 * p1 * r1 / max(1, p1 + r1)

        # class 0: weibo→real, finefake→fake
        p0 = TN / max(1, TN + FN)
        r0 = TN / max(1, TN + FP)
        f1_0 = 2 * p0 * r0 / max(1, p0 + r0)

        metrics_by_category[c].update({
            'precision': round((p0 + p1) / 2, 4),
            'recall': round((r0 + r1) / 2, 4),
            'fscore': round((f1_0 + f1_1) / 2, 4),
            'acc': round((TP + TN) / max(1, TP + TN + FP + FN), 4),
        })
    return metrics_by_category