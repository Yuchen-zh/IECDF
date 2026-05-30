import os
import time
import random
import argparse
import numpy as np
import torch

from dataset.clip_dataloader import bert_data as weibo_data
from dataset.weibo21_clip_dataloader import bert_data as weibo21_data
from dataset.finefake_dataloader import bert_data as finefake_data

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

class Run:
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

    def main(self):

        if "finefake" in self.dataset:
            from model.Net_EN import Trainer as MDTrainer
        else:
            from model.Net_CN import Trainer as MDTrainer

        train_loader, val_loader, test_loader, category_label_distribution = \
            self.get_dataloader()

        trainer = MDTrainer(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            bert=self.bert,
            batchsize=self.batchsize,
            use_cuda=self.use_cuda,
            lr=self.lr,
            train_loader=train_loader,
            dropout=self.dropout,
            weight_decay=self.weight_decay, 
            val_loader=val_loader,
            test_loader=test_loader,
            category_dict=self.category_dict,
            early_stop=self.early_stop,
            epoches=self.epoch,
            save_param_dir=os.path.join(self.save_param_dir, self.model_name),
            category_label_distribution=category_label_distribution,
            domain_num=self.domain_num,
            dataset=self.dataset
        )

        trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='IECDF')
    parser.add_argument('--dataset', default='weibo21')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=197)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--early_stop', type=int, default=45)
    parser.add_argument('--bert_vocab_file', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
    parser.add_argument('--root_path', default='./data/')
    parser.add_argument('--bert', default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--domain_num', type=int, default=6)
    parser.add_argument('--bert_emb_dim', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--emb_type', default='bert')

    parser.add_argument(
        '--save_param_dir',
        default='/HOME/pxyai/pxyaih_0031/Performance01/IECDF/param_model/'
    )

    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d_%H%M%S")
    save_param_dir = os.path.join(args.save_param_dir, f"{time_str}_{args.dataset}")


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
        f'lr: {args.lr}; model name: {args.model_name}; '
        f'emb_type: {args.emb_type}; batchsize: {args.batchsize}; '
        f'epoch: {args.epoch}; gpu: {args.gpu}; emb_dim: {emb_dim}'
    )

    config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model': {'mlp': {'dims': [384], 'dropout': 0.2}},
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': save_param_dir,
        'dataset': args.dataset,
        'domain_num': args.domain_num
    }

    Run(config).main()