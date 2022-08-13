class Config():
    def __init__(self):
        self.folds = 5
        self.task_type = 'FT'
        self.criterion = 'CE' # 'CE/FOCAL'
        self.pretrained_model_path = './models/pretrain/infoxlm-large' # xlm-roberta-large
        self.model_type = 'infoxlm-large'
        self.maxlen = 180
        self.train_bs = int(32 * 4)
        self.eval_every = 100
        self.num_labels = 4
        self.dropout_bertout = 0.2
        self.if_clean = True
        # whether to use fgm for adversial attack in training
        self.use_fgm = False
        # whether to use pgd for adversial attack in training
        self.use_pgd = False
        # whether to use lookahead in training
        self.optimizer_name = ''
        # whether to use noise_tune in training
        self.use_noise = False
        self.noise_rate = 0.15
        # whether to use learning rate scheduler in training
        self.use_scheduler = True
        self.epochs = 4
        self.classifier_lr = 1e-4
        self.weight_decay = 1e-2
        self.adam_epsilon = 1e-6
        self.num_warmup_ratio = 0.1
        self.lr = 1e-5
        self.hidden_size = 1024
        self.eval_bs = 256 # 16gb我觉得可以
        self.log_files = f'./logs/Finetune_modeltype_{self.model_type}_maxlen_{self.maxlen}_fold_{self.folds}.log'
        self.save_dir = '../ckptFineTune/' # 模型路径
        self.data_dir = '../../data/2/trainData.pkl' # 数据路径
        self.test_dir = '../../data/2/testData.pkl' # 数据路径
        # settings for inference
        self.infer_output_dir = './subs/'
        self.infer_bs = self.eval_bs

if __name__ == '__main__':
    config = Config()
