#!/usr/bin/env python

import pandas as pd
import random
import tempfile

from shared.base_predictor import BasePredictor, PathType

###############################
# test code of install package
import torch
import transformers
import lightgbm
import numpy as np
import pandas as pd
import onnx, onnxruntime
print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'lightgbm: {lightgbm.__version__}')
print(f'np: {np.__version__}')
print(f'pd: {pd.__version__}')

print(f'cuda enable: {torch.cuda.is_available()}')
print(f'current_device: {torch.cuda.current_device()}')
print(f'device: {torch.cuda.device(0)}')
print(f'device_count: {torch.cuda.device_count()}')
print(f'get_device_name: {torch.cuda.get_device_name(0)}')
###############################
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
transformers.utils.logging.set_verbosity_error()
import glob
import sys
from time import gmtime, strftime
###############################
"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
3 diff of task 2 and task 3
1. class Task2Predictor(BasePredictor):
2. self.task
3. predictor = Task2Predictor()
4. aicrowd.json!!!!!!!!!!!!
"""
PERFIX = "/opt/conda/lib/python3.8/site-packages/starter_kit/"   #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PERFIX = ""                                                      #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class Task2Predictor(BasePredictor):           #<<<<<<<<<<<<<<<<<<<
# class Task3Predictor(BasePredictor):             #<<<<<<<<<<<<<<<<<<<
    def prediction_setup(self):
        print("nowtime: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        self.task = 2       #2 or 3   #<<<<<<<<<<<<<<<<<<<
#         self.task = 3                 #<<<<<<<<<<<<<<<<<<<
        self.ALLANSWERPOOLS = {}      # save all results!!!!!!!!!!!!!

    def predict(self,
                test_set_path: PathType,
                product_catalogue_path: PathType,
                predictions_output_path: PathType,
                register_progress=lambda x: print("Progress : ", x)):        
        testDF = pd.read_csv(test_set_path)
        testDF = testDF.head(400).reset_index(drop=True)   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print(f"dataset size {testDF.shape}")
        productDF = pd.read_csv(product_catalogue_path) 
        testData = pd.merge(testDF, productDF, left_on=['product_id','query_locale'],right_on=['product_id','product_locale'],how='left')
        del testDF, productDF; gc.collect()
        ALLSAMPLENUM = len(testData)
 ################################################################################################################ 
        # 5fold info140
#         from .code.zlh.info140.config import Config as info140Config                      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
#         from .code.zlh.info140.data_utils import InferDataSet as info140InferDataSet      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
        from code.zlh.info140.config import Config as info140Config                     # xianxia<<<<<<<<<<<<<<<<<<<<<<<<<<
        from code.zlh.info140.data_utils import InferDataSet as info140InferDataSet     # xianxia<<<<<<<<<<<<<<<<<<<<<<<<<<
        pool_model_paths_info140 = glob.glob( PERFIX+"./models/ourModel/info140/*onnx" )  # xianshang<<<<<<<<<
        BSZ=256
        config = info140Config()
        ans = {'example_id':[],'esci_logits':[]} 
        tokenizer = AutoTokenizer.from_pretrained( PERFIX+config.pretrained_model_path )
        T2ValData = info140InferDataSet(testData, tokenizer, config.maxlen, if_clean=config.if_clean)
        print(f"zlh config:\nmaxlen:{config.maxlen}; BSZ:{BSZ}; models: {pool_model_paths_info140}")
        ValDataloader = torch.utils.data.DataLoader(T2ValData, batch_size=BSZ, num_workers=8, drop_last=False, shuffle=False)
        for nowidx, now_model_path in enumerate(pool_model_paths_info140):
            session = onnxruntime.InferenceSession(now_model_path,providers=["CUDAExecutionProvider"]) 
            print(f"#####{now_model_path} start at: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            tmpans = {'example_id':[],'esci_logits':[]}
            for batchID, data in enumerate(ValDataloader):
                inputs = data['Inputs'].numpy()
                outputs = session.run(None, {'inputs': inputs})
                tmpans['esci_logits'].extend(outputs)
                tmpans['example_id'].extend(data['exampleID'])
                this_model_progress = ((batchID + 1) / (ALLSAMPLENUM//BSZ)) / 8 + nowidx / 8
                register_progress(this_model_progress)         # do not delete this line
            
            if len(ans['example_id'])==0:              #  directly mean 
                ans['example_id'] = np.vstack(tmpans['example_id']).flatten()
                ans['esci_logits']= np.vstack(tmpans['esci_logits'])
            else:
                assert (ans['example_id'] == np.vstack(tmpans['example_id']).flatten()).all()
                ans['esci_logits'] += np.vstack(tmpans['esci_logits'])
            del session, tmpans, inputs, outputs; gc.collect()
            print("===> finished tmpans infer at: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )

        ans['esci_logits'] = np.array(ans['esci_logits']) / len(pool_model_paths_info140)
        self.ALLANSWERPOOLS["info140"] = ans
        del ans; gc.collect()
 ################################################################################################################ 
        # 5fold info180
#         from .code.zlh.info180.config import Config as info180Config                      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
#         from .code.zlh.info180.data_utils import InferDataSet as info180InferDataSet      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
        from code.code1.info180.config import Config as info180Config                     # xianxia<<<<<<<<<<<<<<<<<<<<<<<<<<
        from code.code1.info180.data_utils import InferDataSet as info180InferDataSet     # xianxia<<<<<<<<<<<<<<<<<<<<<<<<<<
        pool_model_paths_info180 = glob.glob( PERFIX+"./models/ourModel/info180/*onnx" )  # xianshang<<<<<<<<<
        BSZ=256
        config = info180Config()
        ans = {'example_id':[],'esci_logits':[]} 
        tokenizer = AutoTokenizer.from_pretrained( PERFIX+config.pretrained_model_path )
        T2ValData = info180InferDataSet(testData, tokenizer, config.maxlen, if_clean=config.if_clean)
        print(f"zlh config:\nmaxlen:{config.maxlen}; BSZ:{BSZ}; models: {pool_model_paths_info180}")
        ValDataloader = torch.utils.data.DataLoader(T2ValData, batch_size=BSZ, num_workers=8, drop_last=False, shuffle=False)
        for nowidx, now_model_path in enumerate(pool_model_paths_info180):
            session = onnxruntime.InferenceSession(now_model_path,providers=["CUDAExecutionProvider"]) 
            print(f"#####{now_model_path} start at: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            tmpans = {'example_id':[],'esci_logits':[]}
            for batchID, data in enumerate(ValDataloader):
                inputs = data['Inputs'].numpy()
                outputs = session.run(None, {'inputs': inputs})
                tmpans['esci_logits'].extend(outputs)
                tmpans['example_id'].extend(data['exampleID'])
                this_model_progress = ((batchID + 1) / (ALLSAMPLENUM//BSZ)) / 8 + nowidx / 8 + 0.25
                register_progress(this_model_progress)         # do not delete this line
            
            if len(ans['example_id'])==0:              #  directly mean 
                ans['example_id'] = np.vstack(tmpans['example_id']).flatten()
                ans['esci_logits']= np.vstack(tmpans['esci_logits'])
            else:
                assert (ans['example_id'] == np.vstack(tmpans['example_id']).flatten()).all()
                ans['esci_logits'] += np.vstack(tmpans['esci_logits'])
            del session, tmpans, inputs, outputs; gc.collect()
            print("===> finished tmpans infer at: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )

        ans['esci_logits'] = np.array(ans['esci_logits']) / len(pool_model_paths_info180)
        self.ALLANSWERPOOLS["info180"] = ans
        del ans; gc.collect()
 ################################################################################################################ 
        # us112
#         from .code.code2.newnewus112.config import Config as us112Config                      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
#         from .code.code2.newnewus112.data_utils import InferDataSet as us112InferDataSet      # xianshang<<<<<<<<<<<<<<<<<<<<<<<<
        from code.code2.newnewus112.config import Config as us112Config                      # xianxia<<<<<<<<<<<<<<<<<<<<<<<<
        from code.code2.newnewus112.data_utils import InferDataSet as us112InferDataSet      # xianxia<<<<<<<<<<<<<<<<<<<<<<<<
        pool_model_paths_us112 = glob.glob( PERFIX+"./models/ourModel/us112/*onnx" )      # xianshang<<<<<<<<<
#         pool_model_paths_us112 = [x for x in pool_model_paths_us112 if ("fold_0" in x) or ("fold_2" in x)]  # xianshang<<<<<<<<<
        BSZ=256
        config = us112Config()
        ans = {'example_id':[],'esci_logits':[]} 
        tokenizer = AutoTokenizer.from_pretrained( PERFIX+config.pretrained_model_path )
        T2ValData = us112InferDataSet(testData, tokenizer, config.maxlen, if_clean=config.if_clean)
        print(f"zlh config:\nmaxlen:{config.maxlen}; BSZ:{BSZ}; models: {pool_model_paths_us112}")
        ValDataloader = torch.utils.data.DataLoader(T2ValData, batch_size=BSZ, num_workers=8, drop_last=False, shuffle=False)
        for nowidx, now_model_path in enumerate(pool_model_paths_us112):
            session = onnxruntime.InferenceSession(now_model_path,providers=["CUDAExecutionProvider"]) 
            print(f"#####{now_model_path} start at: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            tmpans = {'example_id':[],'esci_logits':[]}
            for batchID, data in enumerate(ValDataloader):
                inputs = data['Inputs'].numpy()
                outputs = session.run(None, {'inputs': inputs})
                tmpans['esci_logits'].extend(outputs)
                tmpans['example_id'].extend(data['exampleID'])
                this_model_progress = ((batchID + 1) / (ALLSAMPLENUM//BSZ)) / 8 + nowidx / 8 + 0.5
                register_progress(this_model_progress)         # do not delete this line
            
            if len(ans['example_id'])==0:              #  directly mean 
                ans['example_id'] = np.vstack(tmpans['example_id']).flatten()
                ans['esci_logits']= np.vstack(tmpans['esci_logits'])
            else:
                assert (ans['example_id'] == np.vstack(tmpans['example_id']).flatten()).all()
                ans['esci_logits'] += np.vstack(tmpans['esci_logits'])
            del session, tmpans, inputs, outputs; gc.collect()
            print("===> finished tmpans infer at: " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) )

        ans['esci_logits'] = np.array(ans['esci_logits']) / len(pool_model_paths_us112)
        self.ALLANSWERPOOLS["us112"] = ans
        del ans; gc.collect()
        
 ################################################################################################################ 
        from .code.code2 import usmodel_old128 as zp1usmodel        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< online code
        # from code.code2 import usmodel_old128 as zp1usmodel       # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< debug code
        print("===> us single model nowtime: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        pool_model_paths_ous128 = glob.glob( PERFIX+"./models/ourModel/zp_us1/*onnx" ) # xianshang<<<<<<<<<
        zpcfg = zp1usmodel.CFG()
        zpcfg.tokenizer = AutoTokenizer.from_pretrained(PERFIX+zpcfg.model)
        ustest = zp1usmodel.processdflcy2zp(testData)
        test_dataset = zp1usmodel.TestDataset(zpcfg, ustest)
        BSZ = 128                           # <<<<<<<<<<<<<<<<<<
        print(f"zp config:\n BSZ:{BSZ}; models: {pool_model_paths_ous128}")
        ValDataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BSZ, shuffle=False, num_workers=8, pin_memory=True, drop_last=False) 
        ans = {'example_id':[],'esci_logits':[]} 
        for nowidx2, now_model_path in enumerate(pool_model_paths_ous128):   ##### k fold model
            session = onnxruntime.InferenceSession(now_model_path,providers=["CUDAExecutionProvider"]) 
            print(f"#####{now_model_path} start at: "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            tmpans = {'example_id':[],'esci_logits':[]}
            for batchID, data in enumerate(ValDataloader):
                inputs = data['Inputs'].numpy()
                outputs, _ = session.run(None, {'inputs': inputs})
                tmpans['esci_logits'].extend(outputs)
                this_model_progress = ((batchID + 1) / (ALLSAMPLENUM//BSZ)) / 8 + nowidx / 8 + 0.75
                register_progress(this_model_progress)         # do not delete this line
            if len(ans['esci_logits'])==0:
                ans['esci_logits']= np.vstack(tmpans['esci_logits'])
            else:
                ans['esci_logits'] += np.vstack(tmpans['esci_logits'])

        ans['esci_logits'] = np.array(ans['esci_logits']) / len(pool_model_paths_ous128)

        ans['example_id'] = self.ALLANSWERPOOLS["info140"]['example_id']
        self.ALLANSWERPOOLS["usold128"] = ans
    
        del ans; gc.collect()
######################################################################################################   
        # lgb model!!!!!!!!!!!!!!!
#         from .code.zp import lgbinf     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        from code.zp import lgbinf    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print("saving!")
        for kkey in ["info140", "info180", "us112", "usold128"]:
            ddf = pd.DataFrame(self.ALLANSWERPOOLS[kkey]['esci_logits'], columns = ["col1", "col2", "col3", "col4"])
            ddf['example_id'] = self.ALLANSWERPOOLS[kkey]['example_id']
            ddf.to_csv(f"task_{self.task}_{kkey}.csv")
        print("reading!")
        productDF = pd.read_csv(product_catalogue_path) 
        testDF = pd.read_csv(test_set_path)
        testDF = testDF.head(400).reset_index(drop=True)   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        print("lgb start!")
        sumbit_file = lgbinf.lgb_test_predict(zlh_newnew_us112 = self.ALLANSWERPOOLS["us112"]['esci_logits'], 
                                zlh_predict180 = self.ALLANSWERPOOLS["info180"]['esci_logits'], 
                                zlh_predict_140 = self.ALLANSWERPOOLS["info140"]['esci_logits'], 
                                zp_us_oof2_128 = self.ALLANSWERPOOLS["usold128"]['esci_logits'], 
                                productDF = productDF, 
                                subDF = testDF, 
                                task=self.task,    KD=False, perfix=PERFIX)

        print("Writing Task-2 Predictions to : ", predictions_output_path)
        if self.task==2:
            sumbit_file[["example_id", "esci_label"]].to_csv(predictions_output_path, index=False, header=True)
        else:
            sumbit_file[["example_id", "substitute_label"]].to_csv(predictions_output_path, index=False, header=True)
        nowtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print("===> finished at: "+nowtime)

if __name__ == "__main__":
    # Instantiate Predictor Class
    predictor = Task2Predictor()      #   <<<<<<<<<<<<<<<<<<<<<<<<<
#     predictor = Task3Predictor()     #   <<<<<<<<<<<<<<<<<<<<<<<<<
    predictor.prediction_setup()
    
    # ################################################3
    # # debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # test_set_path = "../../../../KDD/data/2/test_public-v0.3.csv.zip"
    # product_catalogue_path = '../../../../KDD/data/2/product_catalogue-v0.3.csv.zip'
    # output_file_path = "./finalsub.csv"
    # # Make Predictions
    # predictor.predict(
    #     test_set_path=test_set_path,
    #     product_catalogue_path = product_catalogue_path,
    #     predictions_output_path=output_file_path,
    # )
    # exit()   # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ###############################################3
    
    test_set_path = "./data/task_2_multiclass_product_classification/test_public-v0.3.csv.zip"
    product_catalogue_path = "./data/task_2_multiclass_product_classification/product_catalogue-v0.3.csv.zip"

    # Generate a Random File to store predictions
    with tempfile.NamedTemporaryFile(suffix='.csv') as output_file:
        output_file_path = output_file.name

        # Make Predictions
        predictor.predict(
            test_set_path=test_set_path,
            predictions_output_path=output_file_path
        )
        
        ####################################################################################################
        ####################################################################################################
        ## 
        ## Adding some simple validations to ensure that the generated file has the expected structure
        ## 
        ####################################################################################################
        ####################################################################################################
        # Validating sample submission
        predictions_df = pd.read_csv(output_file_path)
        test_df = pd.read_csv(test_set_path)

        # Check-#1 : Sample Submission has "example_id" and "esci_label" columns
        expected_columns = ["example_id", "esci_label"]
        assert set(expected_columns) <= set(predictions_df.columns.tolist()), \
            "Predictions file's column names do not match the expected column names : {}".format(
                expected_columns)

        # Check-#2 : Sample Submission contains predictions for all example_ids
        predicted_example_ids = sorted(predictions_df["example_id"].tolist())
        expected_example_ids = sorted(test_df["example_id"].tolist())
        assert expected_example_ids == predicted_example_ids, \
            "`example_id`s present in the Predictions file do not match the `example_id`s provided in the test set"

        # Check-#3 : Predicted `esci_label`s are valid
        VALID_OPTIONS = sorted(
            ["exact", "complement", "irrelevant", "substitute"])
        predicted_esci_labels = sorted(predictions_df["esci_label"].unique())
        assert predicted_esci_labels == VALID_OPTIONS, \
            "`esci_label`s present in the Predictions file do not match the expected ESCI Lables : {}".format(
                VALID_OPTIONS)