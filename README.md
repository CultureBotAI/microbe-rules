# microbe-rules

This is accompanying directory for the article.

Folder structure is the following:

- *src* - contains source codes. 
- *outputs* - contains output from a reference run
- *outputs_llm* - contains outputs from LLM on which results were interpreted

To rerun, please follow the files in folder *src*. 
```bash
# Prepare data
python 01_prepare_data_binary.py 

# Run binary models
python 02_compute_compare_models.py 514 --model_id 1 >02_514_model1.txt
python 02_compute_compare_models.py 514 --model_id 2 >02_514_model2.txt
python 02_compute_compare_models.py 514 --model_id 3 >02_514_model3.txt
python 02_compute_compare_models.py 65 --model_id 0 >02_65_model0.txt
python 02_compute_compare_models.py 65 --model_id 1 >02_65_model1.txt
python 02_compute_compare_models.py 65 --model_id 2 >02_65_model2.txt

# Compute feature importance agreement
python 03_compute_feature_importance_agreement.py 65 >03_65.txt
python 03_compute_feature_importance_agreement.py 514 >03_514.txt

# Compute ARA feature importances

python 04_compute_ara.py 65 --trte_set train >04_65_train.txt
python 04_compute_ara.py 65 --trte_set test >04_65_test.txt
python 04_compute_ara.py 514 --trte_set train >04_514_train.txt
python 04_compute_ara.py 514 --trte_set test >04_514_test.txt


```


Note that package was tested with Python 3.12 as CatBoost does not run with Python 3.13