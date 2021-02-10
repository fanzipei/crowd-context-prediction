# crowd-context-prediction

This repository is for the submission of the KDD 2021 ADS.

Currently only the preprocessing and cluster-level prediction code are uploaded. We will update all the codes upon the acceptance of this paper.

The project structure is organized as follows:

- model.py: defines the cluster-level prediction model used in this paper and all baseline models used in this paper.
- config.py: defines the meta parameters in the experiments.
- preprocess_didi_data.ipynb: preprocessing code for DiDi Chengdu data.
- cluster_level_predictor_training{model_name}.ipynb: the training code for each model
- eval_{model_name}.ipynb: the evaluation code for each model
- {model_name}.pytorch: pre-trained model.
- ensemble: the components predictors for ensemble predictor.
