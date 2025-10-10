from cleverminer.clmec import clmec, clmeq_rq
import os

import pandas as pd
from sklearn import metrics
from sklearn.metrics import  confusion_matrix,accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool, cv, EFstrType

from sklearn.model_selection import train_test_split


#select medium here
mediumid = 514
#select model here; valid values are 0,1,2 for medium 65 and 1,2,3 for medium 514
model = 2
data_path = "data"


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
# COMPUTE RULE-BASED CLASSIFIER                                                     #
#####################################################################################


clmpc = None

if mediumid==65:
    if model==0:
        clmpc = clmec(rq_quantifier=clmeq_rq.CONFS, rule_mining_quantifier=clmeq_rq.DBLCONF, rule_mining_quantifier_value=0.5, show_csv_for_export=True, show_processing_details=1,robustness_min_base=3, robustness_min_additioanlly_scored=20)
    elif model==1:
        clmpc = clmec(rq_quantifier=clmeq_rq.CONFS, rule_mining_quantifier=clmeq_rq.DBLCONF, rule_mining_quantifier_value=0.5, show_csv_for_export=True, show_processing_details=1)
    elif model==2:
        clmpc = clmec(rq_quantifier=clmeq_rq.CONF, rule_mining_quantifier=clmeq_rq.CONF, rule_mining_quantifier_value=0.2, show_csv_for_export=True, show_processing_details=1)
    else:
        print("MODEL NOT DEFINED.")
        exit(1)
elif mediumid==514:
    if model==1:
        clmpc= clmec(rq_quantifier=clmeq_rq.CONFS,rule_mining_quantifier=clmeq_rq.DBLCONF,rule_mining_quantifier_value=0.5,show_csv_for_export=True,show_processing_details=1,robustness_min_base=20,robustness_min_additioanlly_scored=50)
    elif model==2:
        clmpc = clmec(rq_quantifier=clmeq_rq.CONFS, rule_mining_quantifier=clmeq_rq.DBLCONF,rule_mining_quantifier_value=0.5, show_csv_for_export=True, show_processing_details=1)
    elif model == 3:
        clmpc = clmec(rq_quantifier=clmeq_rq.CONF, rule_mining_quantifier=clmeq_rq.CONF,rule_mining_quantifier_value=0.2, show_csv_for_export=True, show_processing_details=1)
    else:
        print("MODEL NOT DEFINED.")
        exit(1)
else:
    print("MODEL FOR MEDIUM NOT DEFINED.")
    exit(1)


clmpc.check_full_structure('orig_file:y',y)
clmpc.check_full_structure('orig_file:y_train',y_train)


clmpc.fit(X_train,y_train)
clmpc.describe()



y_pred_clmpc = clmpc.predict(X_test)
clmpc.y_for_inner_eval=y_test
tgt,is_fallback=clmpc.predict_proba(X_test,justvector=True,include_also_fallback=True,use_add_conf=False)

unique_p = list(set(tgt))



print("DONE")

print(f"Unique values {list(set(tgt))}")

total_records = 0
correctly_scored = 0
max_cat_cnt =0
for i in range(len(y_test)):
    total_records+=1
    if y_test.iloc[i]==y_pred_clmpc[i]:
        correctly_scored+=1
    if y_test.iloc[i]==clmpc.most_frequent_val:
        max_cat_cnt+=1

print(f" Correctly scored {correctly_scored}/{total_records} = {(correctly_scored/total_records):.5f}, max cat count {max_cat_cnt}")


for item in unique_p:
    cnt_positive=0
    cnt_total=0
    for i in range(len(tgt)):
        if tgt[i]==item:
            cnt_total+=1
            if y_test.iloc[i]==1:
                cnt_positive+=1
    if not(clmpc.is_multiclass):
        print(f"--- value prob {item:.5f}, success in {cnt_positive:>8} out of {cnt_total:>8}, that is {(cnt_positive/cnt_total):>5f} ")
    else:
        print("Not implemented")
        #todo implement


print("Accuracy CLMPC :",metrics.accuracy_score(y_test, y_pred_clmpc))
print("Accuracy CLMPC :",metrics.classification_report(y_test, y_pred_clmpc))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_clmpc)
plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('clmpc'+str(mediumid)+'_'+str(model)+'.pdf', format='pdf')


#####################################################################################
# COMPUTE CATBOOST                                                                  #
#####################################################################################


X_train=X_train_orig
X_test = X_test_orig
y_train = y_train_orig
y_test = y_test_orig



#CONVERT TO CATEGORY INDEXES:
d = {x: i for i, x in enumerate(sorted(set(y_train)|set(y_test)))}

y_test = [d[x] for x in y_test]
y_train = [d[x] for x in y_train]


train_data = Pool(data=X_train, label=y_train, cat_features=[0])
val_data = Pool(data=X_val, label=y_val, cat_features=[0])
test_data = Pool(data=X_test, label=y_test, cat_features=[0])


model = CatBoostClassifier(random_seed=9759,iterations=100,
                           loss_function="MultiClass",verbose=100)



model.fit(train_data, eval_set=val_data)


# Predict on test data
y_pred_cb = model.predict(test_data)
y_pred_proba_cb = model.predict_proba(test_data)[:,1]  # Probabilities for the positive class

# Print metrics
print("Accuracy CB:", accuracy_score(y_test, y_pred_cb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_cb))

# Predict on train data
y_pred_train_cb = model.predict(train_data)
y_pred_proba_train_cb = model.predict_proba(train_data)[:,1]  # Probabilities for the positive class

# Print metrics
print("Accuracy CB TRAIN:", accuracy_score(y_train, y_pred_train_cb))



# True labels are assumed to be in y_test
cm = confusion_matrix(y_test, y_pred_cb)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('confusion_matrix_cb'+str(mediumid)+'.pdf', format='pdf')
#plt.show()

model2 = CatBoostClassifier(random_seed=9759,iterations=1,
                           loss_function="MultiClass",verbose=100)



model2.fit(train_data, eval_set=val_data)

# Predict on test data
y_pred_cb1 = model2.predict(test_data)
y_pred_proba_cb1 = model2.predict_proba(test_data)[:,1]  # Probabilities for the positive class

# Print metrics
print("Accuracy CB1:", accuracy_score(y_test, y_pred_cb1))
print("\nClassification Report:\n", classification_report(y_test, y_pred_cb1))


model3 = CatBoostClassifier(random_seed=9759,iterations=5,
                           loss_function="MultiClass",verbose=100)



model3.fit(train_data, eval_set=val_data)

# Predict on test data
y_pred_cbsmall = model3.predict(test_data)
y_pred_proba_cbsmall = model3.predict_proba(test_data)[:,1]  # Probabilities for the positive class

# Print metrics
print("Accuracy CBSMALL:", accuracy_score(y_test, y_pred_cbsmall))
print("\nClassification Report:\n", classification_report(y_test, y_pred_cbsmall))



print("FEATURE IMPORTANCE")

feature_importance = model3.get_feature_importance()
feature_names = X_test.columns

# Display feature importance
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}, {importance*1000}")

print("FEATURE IMPORTANCE SHAP")


feature_importance = model3.get_feature_importance(data=train_data,type=EFstrType.ShapValues)
feature_names = X_train.columns

# Display feature importance
for name, importance in zip(feature_names, feature_importance):
    class_names = model.classes_
    print(f"{name}; {sum(importance[0]*1000)/len(importance[0])},{sum(importance[1]*1000)/len(importance[1])};{model.classes_}")



#####################################################################################
# EVALUATE CLASSIFIERS AND OVERLAP                                                  #
#####################################################################################


y_pred_clmpc = [d[x] for x in y_pred_clmpc]


total_records = 0
correctly_scored = 0
max_cat_cnt = 0
for i in range(len(y_test)):
    total_records += 1
    if y_test[i] == y_pred_clmpc[i]:
        correctly_scored += 1
    if y_test[i] == clmpc.most_frequent_val:
        max_cat_cnt += 1

print(f" Correctly scored {correctly_scored}/{total_records} = {(correctly_scored / total_records):.5f}, max cat count {max_cat_cnt}")
print("Accuracy CLMPC:", accuracy_score(y_test, y_pred_clmpc))
print("\nClassification Report:\n", classification_report(y_test, y_pred_clmpc))



cnt_tot=0
cnt_clmpc_ok=0
cnt_cb_ok=0
cnt_cb1_ok=0
cnt_cbsmall_ok=0
cnt_cbmatch=0
cnt_cbmatchok=0

for i in range(len(y_test)):
    if is_fallback[i]==0:
        cnt_tot+=1
        if y_test[i] == y_pred_clmpc[i]:
            cnt_clmpc_ok+=1
        if y_test[i] == y_pred_cb[i]:
            cnt_cb_ok+=1
        if y_test[i] == y_pred_cbsmall[i]:
            cnt_cbsmall_ok+=1
        if y_test[i] == y_pred_cb1[i]:
            cnt_cb1_ok+=1
        if y_pred_cb[i] == y_pred_clmpc[i]:
            cnt_cbmatch+=1
            if y_pred_cb[i]==y_test[i]:
                cnt_cbmatchok+=1


print(f"Out of non-fallback {cnt_tot}, scored ok CLMPC {cnt_clmpc_ok} ({cnt_clmpc_ok/cnt_tot*100:.3f}%), CB {cnt_cb_ok} ({cnt_cb_ok/cnt_tot*100:.3f}%), CB_SMALL {cnt_cbsmall_ok}({cnt_cbsmall_ok/cnt_tot*100:.3f}%), CB1 {cnt_cb1_ok}({cnt_cb1_ok/cnt_tot*100:.3f}%)")
print(f" CLMPC and CB  matches in {cnt_cbmatch} cases ({cnt_cbmatchok} correctly scored cases)")



