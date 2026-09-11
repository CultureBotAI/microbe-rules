from cleverminer.clmec import clmec, clmeq_rq
import os

import pandas as pd
from sklearn import metrics
from sklearn.metrics import  confusion_matrix,accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool, cv, EFstrType

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Import shared utilities
from pipeline_utils import (
    load_preprocessed_data,
    create_train_val_test_split,
    convert_labels_to_catboost_format,
    train_catboost_model,
    evaluate_model,
    evaluate_model_on_train,
    save_confusion_matrix,
    DEFAULT_RANDOM_SEED,
    DEFAULT_CB_SEED
)

#select medium here
mediumid = 65
data_path = "data"


#####################################################################################
# LOAD DATA                                                                         #
#####################################################################################

# Load preprocessed data using shared utility
X, y = load_preprocessed_data(mediumid, data_path)


#####################################################################################
# PREPARE DATA                                                                      #
#####################################################################################

# Create train/val/test splits using shared utility
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, mediumid, random_seed=DEFAULT_RANDOM_SEED
)



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

# Convert labels to CatBoost format using shared utility
y_train_cat, y_val_cat, y_test_cat, label_dict = convert_labels_to_catboost_format(
    y_train, y_val, y_test
)

# Train main CatBoost model (100 iterations)
print("\n" + "="*80)
print("Training CatBoost model with 100 iterations")
print("="*80)
model = train_catboost_model(
    X_train, y_train_cat,
    X_val, y_val_cat,
    iterations=100,
    random_seed=DEFAULT_CB_SEED
)

# Evaluate on test and train data
results_cb = evaluate_model(model, X_test, y_test_cat, model_name="CatBoost-100")
evaluate_model_on_train(model, X_train, y_train_cat, model_name="CatBoost-100")

# Save confusion matrix
save_confusion_matrix(
    y_test_cat, results_cb['predictions'],
    f'confusion_matrix_cb{mediumid}.pdf'
)

# Train model with 1 iteration
print("\n" + "="*80)
print("Training CatBoost model with 1 iteration")
print("="*80)
model2 = train_catboost_model(
    X_train, y_train_cat,
    X_val, y_val_cat,
    iterations=1,
    random_seed=DEFAULT_CB_SEED
)
results_cb1 = evaluate_model(model2, X_test, y_test_cat, model_name="CatBoost-1")

# Train model with 5 iterations (for SHAP analysis)
print("\n" + "="*80)
print("Training CatBoost model with 5 iterations")
print("="*80)
model3 = train_catboost_model(
    X_train, y_train_cat,
    X_val, y_val_cat,
    iterations=5,
    random_seed=DEFAULT_CB_SEED
)
results_cbsmall = evaluate_model(model3, X_test, y_test_cat, model_name="CatBoost-5")

# Create Pool objects for SHAP analysis
# NOTE: SHAP analysis is done on training data (the data the model learned from)
train_data = Pool(data=X_train, label=y_train_cat, cat_features=[0])
test_data = Pool(data=X_test, label=y_test_cat, cat_features=[0])


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
# CALCULATE FEATURE IMPORTANCE VALIDITY                                             #
#####################################################################################

feature_importance_test = model3.get_feature_importance(data=test_data,type=EFstrType.ShapValues)
feature_names_test = X_test.columns

dict1 = dict(zip(feature_names, feature_importance))
dict2 = dict(zip(feature_names_test, feature_importance_test))

common_keys = sorted(list(set(dict1.keys()) & set(dict2.keys())))

if not common_keys:
    print("No common keys found to calculate correlation.")

data_for_df = []
for key in common_keys:
    data_for_df.append({'Key': key, 'Value1': sum(dict1[key][1]*1000)/len(dict1[key][1]), 'Value2': sum(dict2[key][1]*1000)/len(dict2[key][1])})
    print({'Key': key, 'Value1': sum(dict1[key][1]*1000)/len(dict1[key][1]), 'Value2': sum(dict2[key][1]*1000)/len(dict2[key][1])})

merged_df = pd.DataFrame(data_for_df)

values1 = merged_df['Value1'].tolist()
values2 = merged_df['Value2'].tolist()

if len(values1) < 2 or len(values2) < 2:
    print("Not enough common data points (at least 2 required) to calculate correlation.")

# Calculate Pearson correlation
correlation_coefficient, p_value = pearsonr(values1, values2)

print(f"Correlation between SHAP values for TRAIN and TEST is {correlation_coefficient}, p-value is {p_value}")





