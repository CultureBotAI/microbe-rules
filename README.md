# microbe-rules

This is accompanying directory for the article.

Folder structure is the following:

- *src* - contains source codes. 
- *outputs* - contains output from a reference run
- *outputs_llm* - contains outputs from LLM on which results were interpreted

To rerun, please follow the files in folder *src*. Start with a file *01_prepare_data_binary.py* that will ask you to download a data into subfolder *src/* and based on this data, it will do the data preparation.

Then, run *02_compute_compare_models.py* to rerun models. Note that you might select which modeland for which medium to rerun, see first lines.

```
#select medium here
mediumid = 514
#select model here; valid values are 0,1,2 for medium 65 and 1,2,3 for medium 514
model = 2
```

Note that package was tested with Python 3.12 as CatBoost does not run with Python 3.13