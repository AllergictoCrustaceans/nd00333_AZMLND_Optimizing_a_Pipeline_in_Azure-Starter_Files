# Optimizing an ML Pipeline in Azure

I am still learning AzureML (let alone general ML), and ran into a lot of hiccups throughout the project. 
I had to google/github/stackoverflow for some of my answers in this project. In honor of the honor system, I would like to list out who/where I got help to debug/solve my issues:
- [Sanity/debug check from this guy-- thank you](https://github.com/QuirkyDataScientist1978/Microsoft-Azure-Machine-Learning-Engineer-Project-1-Udacity-Solution)
- [How to save a model](https://github.com/MicrosoftDocs/azure-docs/issues/45773)

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

This dataset contains data about subjects from various socio-economic backgrounds, and we seek to predict whether these subjects would agree to take part in a bank's marketing campaign. We choose to tackle this ML problem through logistic regression and classification. 

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing model was with AutoML, where the VotingEnsemble pipeline (iteration 24) yielded an accuracy of 91.6%. 
The accuracy from this autoML model was 0.2% higher than that of the best run from the logistic regression model w/ HyperDrive tuning (which yielded an accuracy of 91.4%). 
Although autoML yielded a slightly better result, the difference in accuracy between the two models may not be extremely significant. 


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The SKLearn pipline uses 'train.py' as its entry script, logistic regression as its ML model of choice, and HyperDrive for its hyperparameter tuning. 
### What train.py handles in the pipeline: 
- Function `clean_data` is defined. It cleans the raw data by dropping NA values, dropping unwanted columns (e.g. job, contact, education), one-hot-encoding categorical values in certain columns (e.g. marital, default, housing, loan, poutcome), and one-hot-encoding the y variable (e.g. column named 'y').
- Function `main` is defined. It imports raw data via `TabularDatasetFactory.from_delimited_files()` method, calls the `clean_data()` to clean the raw data, splits the data into train/test distributions via `sklearn.model_selection.train_test_split()`, creates a Logistic Regression model with parameters C and max_iter., fits the training dataset to it, and predicts the accuracy of the trained model by passing in the testing dataset.  

### What does HyperDrive handle in the pipeline:
Within the Jupyter notebook, the HyperDrive tuning needs to be configured before its run (shown in screenshot below):

![image](https://user-images.githubusercontent.com/50812346/147689359-1080a479-22b5-452e-8dda-92a3e6dc9149.png)

As you can see, some of the parameters set for us to fill in include:
- `hyperparameter_sampling`: the specified hyperparameter sampling space
- `primary_metric_name`: the primary metric to be assessed during experiemntal runs 
- `primary_metric_goal`: The parameter associated to the primary metric (in the case of "Accuracy", we specify whether to maziminize/minimize this metric when evaluating runs)
- `max_total_runs`: the total number of runs 
- `policy`: the early stopping policy to use
- `estimator`: the estimator that's calleda long with sampled hyperparameters 

### What else is a part of the pipeline:
- Run the hyperdrive model via calling `exp.submit()` and passing in the hyperdrive configurations into the method. 
- After training and testing the model, we call `get_best_run_by_primary_metric` to fetch the best run model, and `register_model()` to save the best run model. 

**What are the benefits of the parameter sampler you chose?**
I chose to use the default sampler that was already imported into the notebook cell, which was the RandomParameterSampling sampler. Since this sampler enables random selection of hyperparameters (discrete and/or continuous), this adds the benefit of assessing a wide range of values that would help yield the best performing model. This reduces a lot of guess work on the ML Engineers side too.

**What are the benefits of the early stopping policy you chose?**
I chose to use the BanditPolicy early stopping policy, which was already imported by deault into the notebook. This policy terminates any runs where the primary metric (e.g. accuracy) does not meet the bounds of the best performing run's slack factor. The BanditPolicy ensures us to return models that are better than the last best performing one, and would not waste time running iterations on ones that aren't performing as well. So, all in all, the benefit of this early stopping policy is that we are guaranteed to get the best performing model, since it is actively looking for models that surpass that of the last best performing models' metrics' threshold. 

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML needs to be configured before it's executed to run.(as shown in the screenshot below): 

![image](https://user-images.githubusercontent.com/50812346/147689252-db51d1cb-3f38-460b-b4c3-0903027bd623.png)

As you can see, some of the parameters set for us to fill in include:
- `experiemnt_timeout_minutes`: allowing autoML to run for no more than the allotted time
- `task`: the type of task this autoML experiment will do
- `primary_metric`: the type of metric that this autoML experiemnt will optimize its model selection for
- `training_data`: the training data (features only) used for this autoML experiment
- `label_column_name`: the column name (output only) that needs to be predicted in this autoML experiment
- `n_cross_validations`: the number of cross validations to perform when user validation data isn't specified
- `enable_early_stopping`: boolean that allows early stopping when training score hasn't improved
- `enable_onnx_compatible_models`: boolean that ensures models to be converted as onnx models

Most of the models its generated has a best metric of 91.47%, with Voting Ensemble as the only model yielding a 91.60% (shown in the screenshot below):

![image](https://user-images.githubusercontent.com/50812346/147696722-9485dd55-3543-4bba-b166-a59de6f52865.png)

As you can see, autoML generates a series of models throughout its run, self-evaluating which model yields the best metrics. This eliminates the time an ML Engineer would normally spend, of manually setting and tuning parameters one training model at a time.

It also outputs which features most impactful in training the best model. 

![image](https://user-images.githubusercontent.com/50812346/147692670-40803f49-a4b5-4e3b-9cf5-1aabe81334c7.png)

In our case, it seems like `duration` makes up for than 50% of feature importance.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

In terms of accuracy, (also -- as stated in the `Summary` section of this README.md) the AutoML's best model came from the VotingEnsemble pipeline, yielding an accuracy of 91.6%, compared to that of logistic regression w/ HyperDrive tuning which yielded an accuracy of 91.4%. In terms of architecture, the SKLearn pipeline only trains and tests its data on logistic regression, while AutoML trains and tests its data on multiple models. Overall, I am unsure of how significant a 0.2% difference is between the two models. However, since AutoML results faired better, I am assuming it is because of its ability to assess different kinds of models' accuracy in a short amount of time, rather than just that of logistic regression.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

While model parameter tuning is important, collecting and preprocessing useful data is just as, if not more, important. I would focus more on defining our data properties clearly, cleaning our data in a consistent manner, as I think this is what makes/breaks a good model.

In terms of data collection and preprocessing, -- and if we were to stick with the same dataset -- I think collecting more data would serve to train our model better. Also, mitigating class imbalance within the data would help reduce bias in the model, therefore producing a model that can predict better. On the other hand, if we are choosing to redefine our dataset, I would research what other features we have not yet accounted for, that may be helpful in predicting whether a subject will part take in a bank marketing campaign. For example, we can always collect surveyed information of whether subjects are in good standing with banks (e.g. have a financial career, have good credit score, related to financiers, major shareholders of banks, etc.), and if this would affect their decision in partaking on a bank marketing campaign.

In terms of model training, I am curious what a bigger dataset and a longer runtime would do in training the model. Since autoML's run process is quite comprehensive, I would utilize autoML to the max, and see what better results it can yield given that we've defined our dataset features/values more clearly. As for training our model with HyperDrive tuning, we can try setting different parameter samplers, and different early stopping policies most optimal to logistic regression to see what better results they yield as well.

## Proof of cluster clean up

(I hoped on the VM again to run this cell, just for you!)

![image](https://user-images.githubusercontent.com/50812346/147692226-b9a56dc8-491f-4957-9341-6cbe5a47db07.png)

