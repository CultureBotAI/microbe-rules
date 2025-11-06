

import math

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from numpy import argmax
from numpy import arange
import sys
from sklearn.metrics import f1_score
from datetime import datetime

import scipy
import sklearn.impute
#from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from cleverminer import cleverminer
import pickle
import operator
import openpyxl

from catboost import CatBoostClassifier, Pool, cv, EFstrType

import os
import argparse
data_path = "data"

#####################################################################################
# PARSE ARGUMENTS
#####################################################################################


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run model training with a specified medium ID.')
    parser.add_argument('mediumid', type=int, help='The medium ID for model training.')

    parser.add_argument('--trte_set', type=str, help='TRAIN or TEST set')


    return parser.parse_args()


args = parse_arguments()

mediumid = args.mediumid

trte_set = args.trte_set

if not(trte_set=='train') and not(trte_set=='test'):
    print("Invalid trte_set ({trte_set}). Please use only values 'train' or 'test'")
    exit(1)

print("******************************************************************************")
print(f"*** Parameters:")
print(f"*** Medium ID  : {mediumid}")
print(f"*** Set        : {trte_set}")
print("******************************************************************************")



#####################################################################################
# LOAD DATA                                                                         #
#####################################################################################

modellabel = "binary_permute_" + str(mediumid)
file_path = 'taxa_to_media__' + str(modellabel) + '_data_df_clean.tsv.gz'
file_path = os.path.join(data_path, file_path)
data_df_clean = pd.read_csv(file_path, sep='\t')
print(f"Dataset has {len(data_df_clean.index)} rows and {len(data_df_clean.columns)} columns")
#exit(458)

# Splitting the data into features and target labels
X = data_df_clean.drop('medium', axis=1)
X = X.drop('subject', axis=1)
y = data_df_clean['medium']


#####################################################################################
# PREPARE DATA                                                                      #
#####################################################################################


y = y.replace(['other','medium:'+str(mediumid)],[0,1])


RANDOM_SEED=12
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp,
                                                random_state=RANDOM_SEED)



X_train_orig=X_train.copy(deep=True)
X_test_orig = X_test.copy(deep=True)
y_train_orig = y_train.copy(deep=True)
y_test_orig = y_test.copy(deep=True)


X=pd.concat([X_train,X_test])
y=pd.concat([y_train,y_test])

#####################################################################################
# COMPUTE ARA - RULE-BASED FEATURE IMPORTANCE                                       #
#####################################################################################


from araxai import ara

do_ara=1

if do_ara>0:

    df=X_train
    df['target']=y_train
    
    mb=5
    
    if trte_set=='test':
        df=X_test
        df['target']=y_test
        mb=1

    print(f"COLUMNS: {df.columns}, ROWS: {len(df.index)}")


    a = ara(df=df,target='target',target_class=1,options={"min_base":mb,"max_depth":1,"boundaries":[1,10,15],"font_size_normal":8})


    #print text results
    a.print_result()

    #print task summary
    a.print_task_info()

    #print run statistics
    a.print_statisics()

    #export charts/results
    a.draw_result()

    res = a.res

    print(res)

    rules = res['results']['rules']

    df_csv=[]

    def _gen_text(r):
        da = []
        d={}
        o=1
        o_max=o
        for v in r['vars']:
            d['varname'+str(o)]=v['varname']
            d['varval'+str(o)]=v['values_str']
            if o>o_max:
                o_max=o
            o = o + 1
        d['lift']=r['lift']
        d['cumlift']=r['cumlift']
        d['booster']=r['booster']
        da.append(d)
        if len(r['sub'])>0:
            for s in r['sub']:
                dd,om = _gen_text(s)
                if om>o_max:
                    o_max=om
                da.extend(dd)

        return da,o_max

    d={}
    h_cnt=0
    for r in rules:
        d,o_max = _gen_text(r)
        if o_max>h_cnt:
            h_cnt=o_max
        df_csv.extend(d)

    h=[]
    for l in range(h_cnt):
        h.append('varname'+str(l+1))
        h.append('varval'+str(l+1))
    h = h+['lift','cumlift','booster']
    print(h)
    df=pd.DataFrame(df_csv)
    df = df[h]
    df.to_csv(f"5summary_ara_{mediumid}_{trte_set}.csv")

