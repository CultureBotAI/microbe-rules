  python src/kg_microbe_train_binary_medium__pipeline.py 65 \
      --data traits \
      --closure false \
      --cv_folds 0 \
      --n_samples 0 \
      --taxonomic-stratify \
      --taxonomic-level family \
      --config src/kg_microbe_train__config_local_rule_mining.json


  python src/kg_microbe_train_binary_medium__pipeline.py 65 \
      --data traits \
      --closure false \
      --cv_folds 0 \
      --n_samples 0 \
      --taxonomic-stratify \
      --taxonomic-level genus \
      --config src/kg_microbe_train__config_local_rule_mining.json


  python src/kg_microbe_train_binary_medium__pipeline.py 514 \
      --data traits \
      --closure false \
      --cv_folds 0 \
      --n_samples 0 \
      --taxonomic-stratify \
      --taxonomic-level family \
      --config src/kg_microbe_train__config_local_rule_mining.json


  python src/kg_microbe_train_binary_medium__pipeline.py 514 \
      --data traits \
      --closure false \
      --cv_folds 0 \
      --n_samples 0 \
      --taxonomic-stratify \
      --taxonomic-level genus \
      --config src/kg_microbe_train__config_local_rule_mining.json
