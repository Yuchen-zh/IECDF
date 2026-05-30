# -*-codeing = utf-8 -*-
import pickle
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaTokenizer
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import re
import numpy as np
from PIL import Image
from pre_train_models.CLIP_BERT import clip

MAX_TEXT_LENGTH = 77

def _init_fn(worker_id):
    np.random.seed(2025)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def text_preprocessing(text):
    if isinstance(text, bytes):
        text = text.decode('utf-8')  # 从字节解码为字符串
    elif not isinstance(text, str):
        raise ValueError("Input must be a string or bytes-like object")
    
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    text = re.sub(r'&amp;', '&', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def word2input(texts, max_len):
    tokenizer = RobertaTokenizer.from_pretrained('/HOME/pxyai/pxyaih_0031/Performance01/IECDF/roberta-base')
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

def process_text(content, max_length):
    processed_content = []
    for text in content:
        if len(text) > max_length:
            text = text[:max_length]
        processed_content.append(text)
    return processed_content

class bert_data():
    def __init__(self,max_len, batch_size, bert_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.category_dict = category_dict
        self.bert_file = bert_file

    def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
        self.data = pd.read_excel(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        content = self.data['text'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['topic'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = self.pre_processing_BERT(content)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        #content = process_text(content, MAX_TEXT_LENGTH)
        clip_text = clip.tokenize(content, truncate=True)
        print("token_ids",token_ids.size())
        print("masks", masks.size())
        print("label", label.size())
        print("category", category.size())
        print("ordered_image", ordered_image.size())
        print("clip_image", clip_image.size())
        print("clip_text", clip_text.size())
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                ordered_image,
                                clip_image,
                                clip_text
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = 0,
            pin_memory = True,
            shuffle = shuffle,
            worker_init_fn = _init_fn,
        )
        
        category_label_distribution = self.get_category_label_distribution(label, category)

        return dataloader, category_label_distribution
    
    def get_category_label_distribution(self, labels, categories):
        category_label_count = {}
        
        for i in range(6):
            i_indices = (categories == i).nonzero(as_tuple=False).squeeze()
            
            i_labels = labels[i_indices]
            
            label_0_count = torch.sum(i_labels == 0).item()
            label_1_count = torch.sum(i_labels == 1).item()
            
            category_label_count[i] = {'label_0': label_0_count, 'label_1': label_1_count}

        print(category_label_count)
        
        return category_label_count

    def pre_processing_BERT(self, texts):
        tokenizer = BertTokenizer.from_pretrained(self.bert_file, do_lower_case=True)
        input_ids = []
        attention_mask = []
        for statement in texts:
            encoded_texts = tokenizer.encode_plus(
                text=text_preprocessing(statement),
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                return_attention_mask=True,
                truncation=True
                )
            
            input_ids.append(torch.tensor(encoded_texts.get('input_ids')))
            attention_mask.append(torch.tensor(encoded_texts.get('attention_mask')))

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        
        return input_ids, attention_mask

