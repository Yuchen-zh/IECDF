import os
import time
import random
import argparse
import numpy as np
import torch
import tqdm

from dataset.clip_dataloader import bert_data as weibo_data
from dataset.weibo21_clip_dataloader import bert_data as weibo21_data
from dataset.finefake_dataloader import bert_data as finefake_data
from utils.utils import clipdata2gpu, metricsTrueFalse

FINEFAKE_CATEGORY = {
    "Politics": 0, "Entertainment": 1, "Business": 2,
    "Health": 3, "Society": 4, "Conflict": 5
}

DATASET_CONFIG = {
    "weibo": {
        "root": "/HOME/pxyai/pxyaih_0031/Performance01/IECDF/data/weibo/",
        "train": "train_origin.csv",
        "val": "val_origin.csv",
        "test": "test_origin.csv",
        "loader": weibo_data,
        "category_dict": {
            "经济": 0, "健康": 1, "军事": 2, "科学": 3,
            "政治": 4, "国际": 5, "教育": 6, "娱乐": 7, "社会": 8
        }
    },

    "weibo21": {
        "root": "/HOME/pxyai/pxyaih_0031/Performance01/IECDF/data/weibo21/",
        "train": "train_datasets.xlsx",
        "val": "val_datasets.xlsx",
        "test": "test_datasets.xlsx",
        "loader": weibo21_data,
        "category_dict": {
            "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3,
            "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8
        }
    }
}

for ratio in ["0.02", "0.05", "0.1", "0.2", "0.3"]:
    DATASET_CONFIG[f"finefake_{ratio}"] = {
        "root": f"/HOME/pxyai/pxyaih_0031/Performance01/IECDF/data/finefake/finefake_{ratio}/",
        "train": f"FineFake_{ratio}_train.xlsx",
        "val": "FineFake_val.xlsx",
        "test": "FineFake_test.xlsx",
        "loader": finefake_data,
        "category_dict": FINEFAKE_CATEGORY
    }


class Tester:
    def __init__(self, config):
        self.__dict__.update(config)

        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']

        cfg = DATASET_CONFIG[self.dataset]

        self.root_path = cfg["root"]
        self.train_path = os.path.join(self.root_path, cfg["train"])
        self.val_path = os.path.join(self.root_path, cfg["val"])
        self.test_path = os.path.join(self.root_path, cfg["test"])

        self.category_dict = cfg["category_dict"]
        self.loader_class = cfg["loader"]

        if "finefake" in self.dataset:
            self.parent_dir = os.path.normpath(os.path.join(self.root_path, ".."))

    def build_loader(self):

        kwargs = {
            "max_len": self.max_len,
            "batch_size": self.batchsize,
            "category_dict": self.category_dict,
            "num_workers": self.num_workers
        }

        if "finefake" in self.dataset:
            kwargs["bert_file"] = self.bert
        else:
            kwargs["vocab_file"] = self.vocab_file

        return self.loader_class(**kwargs)

    def build_cache_path(self, split):

        if "finefake" in self.dataset:

            if split == "train":
                prefix = self.dataset.replace("finefake_", "FineFake_")

                return (
                    os.path.join(self.root_path, f"{prefix}_train.pkl"),
                    os.path.join(self.root_path, f"{prefix}_train_clip.pkl")
                )

            return (
                os.path.join(self.parent_dir, f"FineFake_{split}.pkl"),
                os.path.join(self.parent_dir, f"FineFake_{split}_clip.pkl")
            )

        return (
            os.path.join(self.root_path, f"{split}_loader.pkl"),
            os.path.join(self.root_path, f"{split}_clip_loader.pkl")
        )

    def load_split(self, loader, split, path, shuffle):
        pkl_path, clip_pkl_path = self.build_cache_path(split)
        return loader.load_data(path, pkl_path, clip_pkl_path, shuffle)

    def get_dataloader(self):

        loader = self.build_loader()

        train_loader, category_label_distribution = self.load_split(
            loader, "train", self.train_path, True
        )

        val_loader, _ = self.load_split(
            loader, "val", self.val_path, False
        )

        test_loader, _ = self.load_split(
            loader, "test", self.test_path, False
        )

        return train_loader, val_loader, test_loader, category_label_distribution

    def _build_model(self):

        if "finefake" in self.dataset:
            from model.Net_EN import MDModel
        else:
            from model.Net_CN import MDModel

        model = MDModel(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            bert=self.bert,
            out_channels=320,
            dropout=self.dropout,
            category_dict=self.category_dict,
            domain_num=self.domain_num
        )

        if self.use_cuda:
            model = model.cuda()

        return model

    def _run_inference(self, dataloader, model):

        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label1 = []
        category = []
        labels_by_domain = [[] for _ in range(self.domain_num)]

        model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for _, batch in enumerate(data_iter):
            with torch.no_grad():

                batch_data = clipdata2gpu(batch)
                batch_size_actual = len(batch_data['content'])

                if batch_size_actual < dataloader.batch_size:
                    batch_needed = dataloader.batch_size - batch_size_actual
                    print(f"当前批次数据量不足，补充 {batch_needed} 条数据")

                    remaining_data_iter = iter(dataloader)

                    additional_data = next(remaining_data_iter)

                    additional_data_batch = {
                        key: value[:batch_needed].cuda() for key, value in zip(
                            ['content', 'content_masks', 'label', 'category', 'image', 'clip_image', 'clip_text'],
                            additional_data
                        )
                    }

                    for key in batch_data:
                        batch_data[key] = torch.cat([batch_data[key]] + [additional_data_batch[key]], dim=0)

                    print(f"补充数据: 额外添加了 {batch_needed} 条数据")

                label = batch_data['label']
                batch_category = batch_data['category']

                for i in range(self.domain_num):
                    domain_mask = (batch_category == i)
                    labels_by_domain[i].append(label[domain_mask])

                final_label_pred_list, fusion_label_pred_list, image_label_pred_list, text_label_pred_list = \
                    model(**batch_data)

                idxs = torch.tensor([index for index in batch_category]).view(-1, 1)
                batch_label_pred0 = final_label_pred_list
                batch_label_pred1 = fusion_label_pred_list
                batch_label_pred2 = image_label_pred_list
                batch_label_pred3 = text_label_pred_list

                batch_label = torch.cat([
                    label[idxs.squeeze() == i]
                    for i in range(self.domain_num)
                ])
                batch_category = torch.cat([
                    batch_category[idxs.squeeze() == i]
                    for i in range(self.domain_num)
                ])
                label1.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(batch_label_pred0.detach().cpu().numpy().tolist())
                pred1.extend(batch_label_pred1.detach().cpu().numpy().tolist())
                pred2.extend(batch_label_pred2.detach().cpu().numpy().tolist())
                pred3.extend(batch_label_pred3.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        print("\nPer-domain sample counts:")
        for i in range(self.domain_num):
            labels_i = torch.cat(labels_by_domain[i], dim=0) if labels_by_domain[i] else torch.tensor([])
            print(f"Domain {i} - Samples: {labels_i.shape[0]}")

        return label1, pred0, pred1, pred2, pred3, category

    def _save_all_metrics(self, all_results, save_path):

        branch_names = ["Final", "Fusion", "Image", "Text"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"Test Results - {self.dataset}\n")
            f.write(f"Weight: {self.weight_path}\n")
            f.write(f"Split: {self.test_split}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for idx, result in enumerate(all_results):
                f.write(f">>> {branch_names[idx]} Branch <<<\n")
                f.write("-" * 40 + "\n")

                reserved_keys = {"acc", "precision", "recall", "metric", "auc", "G_means", "real", "fake"}

                f.write("  [Overall]\n")
                for key in ("acc", "precision", "recall", "metric", "auc", "G_means"):
                    if key in result:
                        val = result[key]
                        f.write(f"    {key}: {val:.4f}\n" if isinstance(val, (int, float)) else f"    {key}: {val}\n")

                f.write("\n  [Real News]\n")
                if "real" in result:
                    for key in ("Accuracy", "precision", "recall", "F1", "specificity", "G_means"):
                        if key in result["real"]:
                            val = result["real"][key]
                            f.write(f"    {key}: {val:.4f}\n" if isinstance(val, (int, float)) else f"    {key}: {val}\n")

                f.write("\n  [Fake News]\n")
                if "fake" in result:
                    for key in ("Accuracy", "precision", "recall", "F1", "specificity", "G_means"):
                        if key in result["fake"]:
                            val = result["fake"][key]
                            f.write(f"    {key}: {val:.4f}\n" if isinstance(val, (int, float)) else f"    {key}: {val}\n")

                f.write("\n  [Per-Category]\n")
                for c_name, c_metrics in sorted(result.items(), key=lambda x: str(x[0])):
                    if c_name in reserved_keys:
                        continue
                    if not isinstance(c_metrics, dict):
                        continue
                    f.write(f"    [{c_name}]\n")
                    for key in ("precision", "recall", "fscore", "acc", "auc"):
                        if key in c_metrics:
                            val = c_metrics[key]
                            f.write(f"      {key}: {val:.4f}\n" if isinstance(val, (int, float)) else f"      {key}: {val}\n")

                f.write("\n")

        print(f"\nAll metrics saved to {save_path}")

    def test(self):

        print(f"Loading model architecture...")
        model = self._build_model()

        print(f"Loading weights from {self.weight_path}...")
        state_dict = torch.load(self.weight_path, map_location='cuda' if self.use_cuda else 'cpu')
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")

        train_loader, val_loader, test_loader, _ = \
            self.get_dataloader()

        if self.test_split == "val":
            dataloader = val_loader
            split_name = "val"
        elif self.test_split == "train":
            dataloader = train_loader
            split_name = "train"
        else:
            dataloader = test_loader
            split_name = "test"

        print(f"Running inference on {split_name} set...")
        label1, pred0, pred1, pred2, pred3, category = self._run_inference(dataloader, model)

        save_dir = os.path.dirname(self.weight_path)
        os.makedirs(save_dir, exist_ok=True)

        results0 = metricsTrueFalse(
            label1, pred0, category, self.category_dict,
            save_dir=save_dir, output_file=f"test_final_{split_name}.txt",
            domain_num=self.domain_num
        )

        all_results = [results0]
        summary_path = os.path.join(save_dir, f"test_summary_{split_name}.txt")
        self._save_all_metrics(all_results, summary_path)

        branch_names = ["Final", "Fusion", "Image", "Text"]
        print("\n" + "=" * 60)
        print(f"TEST RESULTS [{split_name.upper()} SET]".center(60))
        print("=" * 60)
        for idx, result in enumerate(all_results):
            print(f"\n>>> {branch_names[idx]} Branch <<<")
            print(f"  Acc: {result.get('acc', 'N/A'):.4f}" if isinstance(result.get('acc'), (int, float)) else f"  Acc: {result.get('acc', 'N/A')}")
            print(f"  F1 (macro): {result.get('metric', 'N/A'):.4f}" if isinstance(result.get('metric'), (int, float)) else f"  F1 (macro): {result.get('metric', 'N/A')}")
            print(f"  AUC: {result.get('auc', 'N/A'):.4f}" if isinstance(result.get('auc'), (int, float)) else f"  AUC: {result.get('auc', 'N/A')}")
            print(f"  G_means: {result.get('G_means', 'N/A'):.4f}" if isinstance(result.get('G_means'), (int, float)) else f"  G_means: {result.get('G_means', 'N/A')}")
        print("=" * 60 + "\n")

        return all_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='IECDF')
    parser.add_argument('--dataset', default='weibo21')
    parser.add_argument('--max_len', type=int, default=197)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--bert_vocab_file', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
    parser.add_argument('--root_path', default='./data/')
    parser.add_argument('--bert', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--domain_num', type=int, default=6)
    parser.add_argument('--bert_emb_dim', type=int, default=768)
    parser.add_argument('--emb_type', default='bert')
    parser.add_argument('--test_split', default='test',
                        help="Which split to test on: test, val, or train")

    parser.add_argument('--weight_path', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/param_model/20260513_221420_weibo21/IECDF/weibo21_best.pkl', help='Path to the trained model weight file (.pkl)'
    )

    parser.add_argument('--save_param_dir', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/param_model/')

    args = parser.parse_args()

    save_param_dir = os.path.dirname(args.weight_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file

    args.bert = (
        '/HOME/pxyai/pxyaih_0031/Performance01/IECDF/bert-base-uncased'
        if 'finefake' in args.dataset else
        '/HOME/pxyai/pxyaih_0031/Performance01/IECDF/pretrained_model/chinese_roberta_wwm_base_ext_pytorch'
    )

    args.domain_num = (6 if 'finefake' in args.dataset else 9)

    print(
        f'model name: {args.model_name}; emb_type: {args.emb_type}; '
        f'batchsize: {args.batchsize}; gpu: {args.gpu}; '
        f'emb_dim: {emb_dim}; test_split: {args.test_split}'
    )
    print(f'weight_path: {args.weight_path}')

    config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model': {'mlp': {'dims': [384], 'dropout': 0.2}},
        'emb_dim': emb_dim,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': save_param_dir,
        'dataset': args.dataset,
        'domain_num': args.domain_num,
        'weight_path': args.weight_path,
        'test_split': args.test_split,
        'lr': 0.0,
        'epoch': 0,
        'early_stop': 0
    }

    Tester(config).test()
