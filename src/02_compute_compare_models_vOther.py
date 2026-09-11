"""
Model training and comparison script for rule-based and gradient boosting classifiers.

This script trains both CleverMiner (rule-based) and CatBoost (gradient boosting) models
for predicting microbial growth on specific media.

Usage:
    python 02_compute_compare_models.py --medium 514 --model 2
    python 02_compute_compare_models.py --medium 65 --model 0
"""

import argparse
from cleverminer.clmec import clmec, clmeq_rq
import os

import pandas as pd
from sklearn import metrics
from sklearn.metrics import  confusion_matrix,accuracy_score, classification_report
from catboost import CatBoostClassifier, Pool, cv, EFstrType

from sklearn.model_selection import train_test_split

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


def run_model_comparison(mediumid: int, model: int, data_path: str = "data") -> None:
    """
    Run model training and comparison for a specific medium and model configuration.

    Parameters:
    - mediumid: Medium ID (65 or 514)
    - model: Model configuration (0,1,2 for medium 65; 1,2,3 for medium 514)
    - data_path: Path to data directory

    Model configurations:
      Medium 65:
        - Model 0: CONFS with robustness (min_base=3, min_additionally_scored=20)
        - Model 1: CONFS without robustness
        - Model 2: CONF (threshold=0.2)
      Medium 514:
        - Model 1: CONFS with strong robustness (min_base=20, min_additionally_scored=50)
        - Model 2: CONFS without robustness
        - Model 3: CONF (threshold=0.2)
    """

def main(mediumid: int, model: int, data_path: str = "data"):
    """Execute the main model comparison workflow."""

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

# Convert labels to CatBoost format using shared utility
y_train_cat, y_val_cat, y_test_cat, label_dict = convert_labels_to_catboost_format(
    y_train, y_val, y_test
)

# Train main CatBoost model (100 iterations) using shared utility
print("\n" + "="*80)
print("Training CatBoost model with 100 iterations")
print("="*80)
model = train_catboost_model(
    X_train, y_train_cat,
    X_val, y_val_cat,
    iterations=100,
    random_seed=DEFAULT_CB_SEED
)

# Evaluate on test data
results_cb = evaluate_model(model, X_test, y_test_cat, model_name="CatBoost-100")
y_pred_cb = results_cb['predictions']
y_pred_proba_cb = results_cb['probabilities']

# Evaluate on train data (check for overfitting)
evaluate_model_on_train(model, X_train, y_train_cat, model_name="CatBoost-100")

# Save confusion matrix
save_confusion_matrix(
    y_test_cat, y_pred_cb,
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
y_pred_cb1 = results_cb1['predictions']
y_pred_proba_cb1 = results_cb1['probabilities']

# Train model with 5 iterations
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
y_pred_cbsmall = results_cbsmall['predictions']
y_pred_proba_cbsmall = results_cbsmall['probabilities']

# Create Pool objects for SHAP analysis
# NOTE: SHAP analysis is done on training data (the data the model learned from)
train_data = Pool(data=X_train, label=y_train_cat, cat_features=[0])


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


y_pred_clmpc = [label_dict[x] for x in y_pred_clmpc]


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


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and compare rule-based (CleverMiner) and gradient boosting (CatBoost) models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model configurations:
  Medium 65:
    - Model 0: CONFS with robustness (min_base=3, min_additionally_scored=20)
    - Model 1: CONFS without robustness constraints
    - Model 2: CONF quantifier (threshold=0.2)

  Medium 514:
    - Model 1: CONFS with strong robustness (min_base=20, min_additionally_scored=50)
    - Model 2: CONFS without robustness constraints
    - Model 3: CONF quantifier (threshold=0.2)

Examples:
  python 02_compute_compare_models.py --medium 514 --model 2
  python 02_compute_compare_models.py --medium 65 --model 0
        """
    )

    parser.add_argument(
        '--medium', '-m',
        type=int,
        required=True,
        choices=[65, 514],
        help='Medium ID (65 or 514)'
    )

    parser.add_argument(
        '--model',
        type=int,
        required=True,
        help='Model configuration (0,1,2 for medium 65; 1,2,3 for medium 514)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # Validate model selection for the given medium
    if args.medium == 65 and args.model not in [0, 1, 2]:
        raise ValueError(f"For medium 65, model must be 0, 1, or 2 (got {args.model})")
    if args.medium == 514 and args.model not in [1, 2, 3]:
        raise ValueError(f"For medium 514, model must be 1, 2, or 3 (got {args.model})")

    print(f"\n{'='*80}")
    print(f"Model Comparison: Medium {args.medium}, Model {args.model}")
    print(f"{'='*80}\n")

    # Call main function with parsed arguments
    main(args.medium, args.model, args.data_path)



