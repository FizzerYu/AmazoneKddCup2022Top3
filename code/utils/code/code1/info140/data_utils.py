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

def cleanText(text):
    pattern = re.compile(r'<[^>]+>',re.S)
    text = pattern.sub('', text)
    return text

class InferDataSet(Dataset):
    def __init__(self, data, tokenizer, max_len, if_clean=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.if_clean = if_clean
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]    
        text = [sample[['query']].astype(str).values[0],
                sample[['product_brand']].astype(str).values[0],
                sample[['product_color_name']].astype(str).values[0],
                sample[['product_title']].astype(str).values[0],
                sample[['product_bullet_point']].astype(str).values[0],
                sample[['product_description']].astype(str).values[0]
               ]
        if self.if_clean:
            text = [cleanText(str(i)) for i in text]
        else:
            text = [str(i) for i in text]
        text = self.tokenizer.cls_token+f' {self.tokenizer.sep_token} '.join(text)+self.tokenizer.sep_token
        Encoding = self.tokenizer(text,padding='max_length',truncation=True,max_length=self.max_len,return_tensors='pt',add_special_tokens=False)
        exampleID = sample[['example_id']].astype(str).values[0]
#         return {
#             'IDs':torch.squeeze(Encoding['input_ids'],0),
#             'AttMask':torch.squeeze(Encoding['attention_mask'],0),
#             'exampleID':exampleID,
#             'label':-1
#         }
        return {
            'Inputs':torch.cat((Encoding['input_ids'],Encoding['attention_mask']),0),
            'exampleID':exampleID,
            'label':-1
        }