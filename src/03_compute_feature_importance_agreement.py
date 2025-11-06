from cleverminer.clmec import clmec, clmeq_rq
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import  confusion_matrix,accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool, cv, EFstrType

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

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

    return parser.parse_args()


args = parse_arguments()

mediumid = args.mediumid


print("******************************************************************************")
print(f"*** Parameters:")
print(f"*** Medium ID  : {mediumid}")
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
val_data = Pool(data=X_train, label=y_train, cat_features=[0])
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

feature_importance = model.get_feature_importance()
feature_names = X_test.columns

# Display feature importance
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}, {importance*1000}")

print(f" LENGTHS FI: {len(feature_names)}, {len(feature_importance)}")
print(f" DIM {feature_importance.ndim}, SHAPE {feature_importance.shape}")



print("FEATURE IMPORTANCE SHAP")


feature_importance = model.get_feature_importance(data=train_data,type=EFstrType.ShapValues)
feature_names = X_train.columns

print(f" LENGTHS: {len(feature_names)}, {len(feature_importance)}")
print(f" DIM {feature_importance.ndim}, SHAPE {feature_importance.shape}")

feature_importance = np.abs(feature_importance)
feature_importance = np.mean(feature_importance[:,:,:-1], axis=(0, 1))
print(f" DIM2 {feature_importance.ndim}, SHAPE {feature_importance.shape}")


#####################################################################################
# CALCULATE FEATURE IMPORTANCE VALIDITY                                             #
#####################################################################################

feature_importance_test = model.get_feature_importance(data=test_data,type=EFstrType.ShapValues)
feature_names_test = X_test.columns

feature_importance_test = np.abs(feature_importance_test)
feature_importance_test = np.mean(feature_importance_test[:,:,:-1], axis=(0, 1))
print(f" DIM2T {feature_importance_test.ndim}, SHAPE {feature_importance_test.shape}")


dict1 = dict(zip(feature_names, feature_importance))
dict2 = dict(zip(feature_names_test, feature_importance_test))

print(f" LENGTHS: {len(feature_names)}, {len(feature_importance)}")
print(f" LENGTHS TEST: {len(feature_names_test)}, {len(feature_importance_test)}")


common_keys = sorted(list(set(dict1.keys()) & set(dict2.keys())))

if not common_keys:
    print("No common keys found to calculate correlation.")

cnt =0

data_for_df = []
for key in common_keys:
    data_for_df.append({'Key': key, 'Value1': dict1[key]*1000, 'Value2': dict2[key]*1000})
    print({'Key': key, 'Value1': dict1[key]*1000, 'Value2': dict2[key]*1000})
    if dict1[key]*1000>0.01:
        cnt = cnt+1

merged_df = pd.DataFrame(data_for_df)

sorted_df = merged_df.sort_values(by='Value1',ascending = False)

sorted_df.to_csv('shap65.csv')

print("TOP 25+25 FEATURES ACCORDING TO SHAP")

print(sorted_df.head(50))
print("...")
print(sorted_df.tail(50))

print(f" {cnt} values >0")

values1 = merged_df['Value1'].tolist()
values2 = merged_df['Value2'].tolist()

if len(values1) < 2 or len(values2) < 2:
    print("Not enough common data points (at least 2 required) to calculate correlation.")

# Calculate Pearson correlation
correlation_coefficient, p_value = pearsonr(values1, values2)

print(f"Correlation between SHAP values for TRAIN and TEST is {correlation_coefficient}, p-value is {p_value}")





