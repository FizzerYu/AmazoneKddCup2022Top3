import os
import re
import gc
import copy
import random
# import argparse
# import logging
# from collections import defaultdict
# import itertools as it
# from tqdm import tqdm
import numpy as np
import pandas as pd
# from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold
# from sklearn.metrics import f1_score
# from sklearn import metrics
# import joblib
import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
# import torch.distributed as dist
# import torch.optim
# from torch.optim import AdamW
# import torch.utils.data.distributed

torch.backends.cudnn.benchmark = True

# from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers import get_linear_schedule_with_warmup
import transformers
# from torch.optim.optimizer import Optimizer, required

###
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.RandomState(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(42)
transformers.utils.logging.set_verbosity_error()

clean_dict = {'</br>': "",
                  '<br/>': "",
                  '<br>': "",
                  '</b>': "",
                  '<b/>': "",
                  '<b>': "",
                  '</li>': "",
                  '<li/>': "",
                  '<li>': "",
                  '</B>': "",
                  '<B/>': "",
                  '<B>': "",
                  '</p>': "",
                  '<p/>': "",
                  '<p>': "",
                  '</i>': "",
                  '<i/>': "",
                  '<i>': "",
                  '/n': " ",
                  '/t': " ",
                  '☆': "",
                  '✿': "",
                  '✔': "",
                  '★': "",
                  '❥': "",
                  '❤': "",
                  '🐳': "",
                  '●':"",
                   '✅':"",
                   '⛛':""
                  }
def cleanText(text):
    for key, value in clean_dict.items():
        text = text.replace(key, value)
    return text


class InferDataSet(Dataset):
    def __init__(self, data, tokenizer, max_len, if_clean):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.if_clean = if_clean
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]    
        product_id = sample['product_id']
        product_id_class = product_id[0]
        product_id_last = product_id[-1]
        product_id_top7 = product_id[:7]
        if product_id_class in {'0','1','2','3','4','5','6','7','8'}:
            query = sample['query'].lower()+' book'
            product_title = 'book' if pd.isnull(sample['product_title']) else sample['product_title'].lower()+' book'
        else:
            query = sample['query'].lower()
            product_title = '' if pd.isnull(sample['product_title']) else sample['product_title'].lower()
        product_brand = '' if pd.isnull(sample['product_brand']) else sample['product_brand'].lower()
        product_color_name = '' if pd.isnull(sample['product_color_name']) else sample['product_color_name'].lower()
        product_bullet_point = '' if pd.isnull(sample['product_bullet_point']) else sample['product_bullet_point'].lower()
        product_description = '' if pd.isnull(sample['product_description']) else sample['product_description'].lower()
        text = [query,product_id_top7, product_title,product_brand,product_color_name,product_bullet_point[:20],product_description]
        if self.if_clean:
            text = [cleanText(str(i)) for i in text]
        else:
            text = [str(i) for i in text]
        text = self.tokenizer.cls_token+f' {self.tokenizer.sep_token} '.join(text)+self.tokenizer.sep_token
        Encoding = self.tokenizer(text,padding='max_length',truncation=True,max_length=self.max_len,return_tensors='pt',add_special_tokens=False)        
        exampleID = sample[['example_id']].astype(str).values[0]
#         return {
#                 'IDs':torch.squeeze(Encoding['input_ids'],0),
#                 'AttMask':torch.squeeze(Encoding['attention_mask'],0),
#                 'exampleID':exampleID,
#                 'label':-1
#                }

        return {
            'Inputs':torch.cat((Encoding['input_ids'],
                                 Encoding['attention_mask'] ),0),
            'exampleID':exampleID,
            'label':-1}
    