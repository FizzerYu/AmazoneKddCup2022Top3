import os
import re
import gc
import copy
import random
import argparse
import logging
from collections import defaultdict
import itertools as it
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold
from sklearn.metrics import f1_score
from sklearn import metrics
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.optim
from torch.optim import AdamW
import torch.utils.data.distributed
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim.optimizer import Optimizer, required

torch.backends.cudnn.benchmark = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.RandomState(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# transformers.utils.logging.set_verbosity_error()
setup_seed(42)
    
    
# class KDDModelBERTBaselineInfer(nn.Module):

#     def __init__(self, config):
#         super(KDDModelBERTBaselineInfer, self).__init__()
#         self.bert = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model_path,
#                                                                        output_hidden_states=True,
#                                                                        num_labels=config.num_labels)
#         for param in self.bert.parameters():
#             param.requires_grad = True

#     def forward(
#         self, data
#     ):
#         out = self.bert(input_ids=data['IDs'], attention_mask=data['AttMask'], )
#         return {'logits':out.logits,}


class KDDModelBERTBaselineInfer(nn.Module):

    def __init__(self, config):
        super(KDDModelBERTBaselineInfer, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(config.pretrained_model_path,
                                                                       output_hidden_states=True,
                                                                       num_labels=config.num_labels)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
        self, data
    ):
        out = self.bert(input_ids=data[:,0,:],attention_mask=data[:,1,:])
        return out.logits
    