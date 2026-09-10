# microbe-rules

This is accompanying directory for the article.

Folder structure is the following:

- *src* - contains source codes.
- *outputs* - contains output from a reference run
- *data/LLM* - the prompts used to interpret the mined rules with an LLM, and
  the feature tables they take as input

There is no *outputs_llm* folder. The README described one until 2026-09-10;
the LLM prompts are in *data/LLM* and the reference-run outputs in *outputs*.

## Interpreting the rules with an LLM

Use **[`data/LLM/RULE_INTERPRETATION_PROMPT.md`](data/LLM/RULE_INTERPRETATION_PROMPT.md)**.
It is medium-agnostic: fill in its section F and it works for any taxon-medium
link. It asks for relationships *between* features (hierarchical roll-ups,
enzyme-substrate pairs, environment-physiology coupling, and links to the
medium's own ingredients) and for the case to be categorised, which the earlier
prompts did not.

The four per-medium prompts beside it produced the published tables and are kept
for reference:

| file | medium | input |
|---|---|---|
| `LLM_514_Table_8_prompt.txt` | 514 | the mined rules |
| `LLM_65_Table_7_prompt.txt` | 65 | the mined rules |
| `LLM_514_Table_11_prompt.txt` | 514 | features grouped by shared property |
| `LLM_65_Table_10_prompt.txt` | 65 | features grouped by shared property |

The three `*_ingredients_prompt.txt` files are a separate task - working out the
composition of a medium ingredient - not rule interpretation.

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