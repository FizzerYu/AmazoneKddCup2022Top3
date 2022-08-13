import pandas as pd
import numpy as np
import lightgbm as lgb
import string
import re 
def jaccard(x, y):
    if str(y) == 'nan':
        y = 'none'
    x = set(x)
    y = set(y)
    return float(len(x & y) / len(x | y))

def blending_fun(data, alpha=0.5):
    print(data['product_locale'].values)
    zlh_map = {
        0:2,
        1:3,
        2:1,
        3:0,
    }
    zlh_map_r = {
        0:3,
        1:2,
        2:0,
        3:1,
    }
    for i in [0,1,2,3]:
        data[f'blending{i}_alpha@{alpha}'] = alpha * data[f'modeltype_infoxlm-large_maxlen_180_col{zlh_map_r[i]}'].values +\
            (1-alpha) * data[f'pred_{i}']
#         np.where(data['product_locale'].values == 'us', \
#                                              (data[f'pred_{i}'].values * alpha +  (1-alpha) * data[f'modeltype_infoxlm-large_maxlen_180_col{zlh_map_r[i]}'].values )/2, \
#                                              data[f'modeltype_infoxlm-large_maxlen_180_col{zlh_map_r[i]}'].values )
    for i in range(4):
        for j in range(i+1, 4):
            data[f'blending_diff_{i}_{j}_alpha@{alpha}'] = data[f'blending{i}_alpha@{alpha}'] - data[f'blending{j}_alpha@{alpha}']

    blending_logit = data[[f'blending{i}_alpha@{alpha}' for i in [0,1,2,3]]].values
    test_probs = np.argmax(blending_logit, axis=1)
    return test_probs

def get_num_common_words_and_ratio(merge, col):
    # merge data
    merge_ = merge[col]
    merge_.columns = ['q1', 'q2']
    merge_['q2'] = merge_['q2'].apply(lambda x: 'none' if str(x) == 'nan' else x)
    merge_['q1'] = merge_['q1'].apply(lambda x: 'none' if str(x) == 'nan' else x)

    q1_word_set = merge_['q1'].apply(lambda x: x.split(' ')).apply(set).values
    q2_word_set = merge_['q2'].apply(lambda x: x.split(' ')).apply(set).values

    q1_word_len = merge_['q1'].apply(lambda x: len(x.split(' '))).values
    q2_word_len = merge_['q2'].apply(lambda x: len(x.split(' '))).values

    q1_word_len_set = merge_['q1'].apply(lambda x: len(set(x.split(' ')))).values
    q2_word_len_set = merge_['q2'].apply(lambda x: len(set(x.split(' ')))).values

    result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
    result_ratio_q = [result[i] / q1_word_len[i] for i in range(len(q1_word_set))]
    result_ratio_t = [result[i] / q2_word_len[i] for i in range(len(q1_word_set))]

    result_ratio_q_set = [result[i] / q1_word_len_set[i] for i in range(len(q1_word_set))]
    result_ratio_t_set = [result[i] / q2_word_len_set[i] for i in range(len(q1_word_set))]

    return result, result_ratio_q, result_ratio_t, q1_word_len, q2_word_len, q1_word_len_set, q2_word_len_set, result_ratio_q_set, result_ratio_t_set

def get_df_grams(train_sample,values,cols):
    def create_ngram_set(input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def get_n_gram(df, values=2):
        train_query = df.values
        train_query = [[word for word in str(sen).replace("'", '').split(' ')] for sen in train_query]
        train_query_n = []
        for input_list in train_query:
            train_query_n_gram = set()
            for value in range(2, values + 1):
                train_query_n_gram = train_query_n_gram | create_ngram_set(input_list, value)
            train_query_n.append(train_query_n_gram)
        return train_query_n

    train_query = get_n_gram(train_sample[cols[0]], values)
    train_title = get_n_gram(train_sample[cols[1]], values)
    sim = list(map(lambda x, y: len(x) + len(y) - 2 * len(x & y),
                       train_query, train_title))
    sim_number_rate=list(map(lambda x, y:   len(x & y)/ len(x)  if len(x)!=0 else 0,
                       train_query, train_title))
    return sim ,sim_number_rate

def create_sparse_matching(data, train=False, match_list = ["product_title", "product_description", "product_brand", "product_bullet_point", "product_color_name"]):
#     from utils import get_num_common_words_and_ratio, get_df_grams

    # jaccard_sim
    for prefix in match_list:
        data[prefix] = data[prefix].astype(str)
        data[prefix + '_jaccard_sim_query'] = list(map(lambda x, y: jaccard(x, y), data['query'], data[prefix]))
    import Levenshtein
    print('get edict distance:')
    for match_col in match_list: 
        print(f'======>{match_col}')
        data[match_col + 'QTedict_distance_k_pt'] = list(
            map(lambda x, y: Levenshtein.distance(x, y) / (len(x)+1), data['query'], data[match_col]))
        data[match_col + 'QTedict_jaro'] = list(
            map(lambda x, y: Levenshtein.jaro(x, y), data['query'], data[match_col]))
        data[match_col + 'QTedict_ratio'] = list(
            map(lambda x, y: Levenshtein.ratio(x, y), data['query'], data[match_col]))
        data[match_col + 'QTedict_jaro_winkler'] = list(
            map(lambda x, y: Levenshtein.jaro_winkler(x, y), data['query'], data[match_col]))
    prefix = match_col
    data[prefix + 'common_words_k_pt'], \
    data[prefix + 'common_words_k_pt_k'], \
    data[prefix + 'common_words_k_pt_pt'], \
    data[prefix + 'k_len'], \
    data[prefix + 'pt_len'], \
    data[prefix + 'k_len_set'], \
    data[prefix + 'pt_len_set'], \
    data[prefix + 'common_words_k_pt_k_set'], \
    data[prefix + 'common_words_k_pt_pt_set']  = get_num_common_words_and_ratio(data, col=['query', prefix])
    data["n_gram_sim_2"], data["n_gram_sim_pa2"] = get_df_grams(data, 2, ['query', prefix])
#         print(f'save to {filename}')
#         joblib.dump(data, filenam
    return data.fillna(0)

remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()

def judge_language(s):
    # s = unicode(s)   # python2需要将字符串转换为unicode编码，python3不需要
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return 'en'

    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 0:
        result.append('zh')
    if len(res2) <= 0:
        return 'zh'
    
    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return 'ko'

    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return 'ja'
    return ','.join(result)

def get_text_len(data):
    t = ""
    for col in ['query','product_title','product_description','product_bullet_point','product_brand','product_color_name']:
        t += data[col].astype(str)
    data["text_len"] = t.apply(len)
    
def llll(x):
    map_language = {'en': 0, '': 1, 'zh': 2, 'ko': 3, 'ja': 4, 'zh,ja': 5, 'en,ja': 6, 'en,zh,ja': 7, 'en,zh': 8}
    if x in map_language:
        return map_language[x]
    else:
        return -1

def lgb_test_predict(zlh_newnew_us112, zlh_predict180, zlh_predict_140, zp_us_oof2_128, productDF, subDF, task=2, KD=False, perfix=""):
    """
    infoxml_pro: N * 4
    us_prob    : N * 4
    subDF     :
    productDF  :
    """
    zlh_predict = pd.DataFrame({
        "example_id": subDF["example_id"]
    })
     # TODO
    for i in range(4):
        zlh_predict[f'modeltype_HM-DS-FGM08-112-deberta-v3-large_maxlen_112_col{i}'] = zlh_newnew_us112[:, i]
        zlh_predict[f'modeltype_infoxlm-large_maxlen_180_col{i}'] = zlh_predict180[:, i]
        zlh_predict[f'modeltype_HM-140-FGM085-RDrop-infoxlm-large_maxlen_140_col{i}'] = zlh_predict_140[:, i]
        zlh_predict[f'pred_{i}'] = zp_us_oof2_128[:, i]
        
        
    # TODO
    for col in ["product_bullet_point", "product_brand", "product_color_name", "product_description"]:
        productDF[f"col_{col}_LE"] = productDF[col].map(dict(zip(productDF[col].unique(), range(len(productDF[col].unique())))))
    
    for i in range(2, 9):
        t = productDF['product_id'].apply(lambda x:str(x)[:i])
        productDF[f'product_id_{i}'] = t.map(dict(zip(t.unique(), range(t.nunique()))))
    
    
    # TODO
    
    # lgb
    esci_label2num = {'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3}
    num2esci_label = ['exact', 'substitute', 'irrelevant', 'complement']

    test_data = pd.merge(
        subDF,
        productDF,
        how='left',
        left_on=['query_locale', 'product_id'],
        right_on=['product_locale', 'product_id']
    )
    test_data = test_data.merge(zlh_predict, how='left', on='example_id')
    
    # TODO
    test_data["language"] = test_data["query"].apply(judge_language)
    test_data["language"] = test_data["language"].apply(lambda x: llll(x))
    get_text_len(test_data)
    
    # TODO
    for alpha in [0.5]:
        print('alpha===>', alpha)
        blending_fun(test_data, alpha=alpha)
    
    test_data['query_count'] = test_data['query'].map(dict(test_data['query'].value_counts())).fillna(0)
    test_data['query_len'] = test_data['query'].apply(len)
    test_data['title_Len'] = test_data['product_title'].astype(str).apply(len)
    test_data['Amazon Basics'] = test_data['product_title'].astype(str).apply(lambda x: 'amazon' in x.lower()).astype(
        int)
    product_locale_dict = {
        'us': 0, 'es': 1, 'jp': 2
    }
    test_data['product_locale'] = test_data['product_locale'].map(dict(product_locale_dict))
    
    #TODO 不要test_data = 
    create_sparse_matching(test_data, train=False)

    # TODO
    all_stat_funtion = ['skew', 'median', 'max', 'mean', 'sum', 'min']
    all_stat_name = ['skew', 'median', 'max', 'mean', 'sum', 'min']    
    for label in [0, 1, 2, 3]:
        for alpha in [0.5]:
            item = f'blending{label}_alpha@{alpha}'
            for name,i in zip(all_stat_name,all_stat_funtion):
                print(item+"_"+name)
                test_data = test_data.merge(test_data.groupby(["query"])[item].agg(i).reset_index().rename(columns={item:item+"_"+name}),on='query',how='left')
    
    # TODO
    for label in [0, 1, 2, 3]:
        for alpha in [0.5]:
            item = f'blending{label}_alpha@{alpha}'
            print(item)
            test_data['groupby_{}_rank{}'.format(label, item)] = test_data.groupby('query')[item].rank()
            test_data['{}_diff'.format(item)] = test_data[item] - test_data[item+"_median"]
            test_data['{}_diff_mean'.format(item)] = test_data[item] - test_data[item+"_mean"]
            test_data['{}_diff_max'.format(item)] = test_data[item] - test_data[item+"_max"]
            test_data['{}_diff_min'.format(item)] = test_data[item] - test_data[item+"_min"]        
    # TODO   
    drop_feat_by_dist = ['modeltype_infoxlm-large_maxlen_180_col3_max', 'modeltype_infoxlm-large_maxlen_180_col0_min', 'modeltype_infoxlm-large_maxlen_180_col3_median', \
 'modeltype_infoxlm-large_maxlen_180_col3', 'modeltype_infoxlm-large_maxlen_180_col0_median', 'modeltype_infoxlm-large_maxlen_180_col3_mean']
    notUseFea = ['esci_label', 'example_id', 'query', 'product_id', 'query_locale', \
                 'product_title', 'product_description', 'product_bullet_point',
                 'product_brand', 'product_color_name'] + [f'modeltype_infoxlm-base_maxlen_128_col{i}' for i in
                                                           range(4)] + drop_feat_by_dist
    feature_columns = [f for f in test_data.columns if f not in notUseFea]
    label_name = 'esci_label'
    used_feat = feature_columns
    print(used_feat)
    print(len(used_feat))
    folds = 5

    sumbit_file = subDF
    preds = 0
    for fold in range(folds):
        print('lgb predict======>', fold)
        modelfile = perfix + "./models/ourModel/zp_lgb/finallgb_model_task{}_fold_{}.txt".format(task, fold)
        print(modelfile)
        model = lgb.Booster(model_file=modelfile)
        pred = model.predict(test_data[used_feat]) / folds
        print(pred)
        print()
        preds += pred

    if task == 2:
        esci_label2num = {'exact': 0, 'substitute': 1, 'irrelevant': 2, 'complement': 3}
        num2esci_label = ['exact', 'substitute', 'irrelevant', 'complement']
        predictions = np.argmax(preds, axis=1)
        sumbit_file['esci_label'] = predictions
        sumbit_file['esci_label'] = sumbit_file['esci_label'].map(lambda x: num2esci_label[x])
    elif task == 3:
        best_threshold = 0.47368421052631576
        predictions = np.where(preds > best_threshold, 1, 0)
        sumbit_file['substitute_label'] = predictions
        sumbit_file['substitute_label'] = sumbit_file['substitute_label'].map(
            lambda x: {1: "substitute", 0: "no_substitute"}[x])
    else:
        assert ValueError("task only support 2 or 3 in int.")
    return sumbit_file