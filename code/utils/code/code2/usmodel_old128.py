from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import re
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel

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


def processdflcy2zp(df):
    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'',text)

    def cleanText(text):
        for key, value in clean_dict.items():
            text = text.replace(key, value)
        return remove_URL(text)

    def concat(data):
        SEQ_LEN=10000
        str_ans = '[CLS]'+data['query'].astype(str).apply(lambda x: x[:SEQ_LEN])+\
            '[SEP]'+data['product_id'].astype(str).apply(lambda x: x[:6])+'[SEP]'+\
            data['product_brand'].astype(str).apply(lambda x: x[:SEQ_LEN])+\
              data['product_color_name'].astype(str).apply(lambda x: x[:SEQ_LEN])+\
                 data['product_id'].astype(str).apply(lambda x: x[:SEQ_LEN])+\
                    data['product_title'].astype(str).apply(lambda x: x[:SEQ_LEN])+\
                        data['product_bullet_point'].astype(str).apply(lambda x: x[:SEQ_LEN])\
                            + data['product_description'].astype(str).apply(lambda x: x[:SEQ_LEN])
        return str_ans.apply(lambda x: cleanText(x))

#     df = df.query("product_locale=='us'").reset_index(drop=True)
    df['text'] = concat(df)
    return df[["example_id", "text"]]


class CFG:
    wandb=False
#     path = OUTPUT_DIR
    config_path= "./models/ourModel/zp_us1/config.pth"
    debug=False
    apex=True
    print_freq=1000
    num_workers=8
#     expname=EXPNAME
    model="./models/pretrain/deberta-v3-large"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=100
    encoder_lr=1e-5
    decoder_lr=1e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=int(32 * 2)
    fc_dropout=0.1
    target_size=4
    SEQ_LEN=2048
    weight_decay=0.0001
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=4
########
    epochs=3
########
    trn_fold=[0, 1, 2, 3]
    train=True
    max_len = 128
    use_us_only = False

def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           return_tensors='pt',
                           padding="max_length",
                           truncation=True,
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
#     return inputs
    return {
            'Inputs':torch.cat((inputs['input_ids'], 
                                 inputs['token_type_ids'],
                                 inputs['attention_mask'] ),0)}

    
class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs
    
def inference_fn(test_loader, model, device):
    preds = []
    features = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs in test_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            y_preds, feature = model(inputs)
            preds.append(y_preds.to('cpu').numpy())
            features.append(feature.to('cpu').numpy())

    predictions = np.concatenate(preds)
    features = np.concatenate(features)
    
    return predictions, features


# class CustomModel(nn.Module):
#     def __init__(self, cfg, config_path=None, pretrained=False):
#         super().__init__()
#         self.cfg = cfg
#         if config_path is None:
#             self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)            
#         else:
#             self.config = torch.load(config_path)
#         if pretrained:
#             self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
#         else:
#             self.model = AutoModel.from_config(self.config)
#         self.fc_dropout = nn.Dropout(cfg.fc_dropout)
#         self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
#         self._init_weights(self.fc)
#         self.attention = nn.Sequential(
#             nn.Linear(self.config.hidden_size, 512),
#             nn.Tanh(),
#             nn.Linear(512, 1),
#             nn.Softmax(dim=1)
#         )
#         out_shape = self.config.hidden_size
#         self._init_weights(self.attention)
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
    
#     def ave_pool(self, inputs):
#         outputs = self.model(**inputs)
#         last_hidden_states = outputs[0]# batch*seq*dim
#         feature = torch.mean(last_hidden_states, 1)
#         return feature
        

#     def forward(self, inputs):
#         feature = self.ave_pool(inputs)
#         output = self.fc(self.fc_dropout(feature))
#         return output, feature



class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)            
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        out_shape = self.config.hidden_size
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def ave_pool(self, inputs):
        outputs = self.model(input_ids=inputs[:,0,:], token_type_ids=inputs[:,1,:], attention_mask=inputs[:,2,:])
        last_hidden_states = outputs[0]# batch*seq*dim
        feature = torch.mean(last_hidden_states, 1)
        return feature
        

    def forward(self, inputs):
        feature = self.ave_pool(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output, feature


