# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Swarangi Sharad Patil

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When the training model was used without any Exploratory Data Analysis or changes to hyperparameters, the model did not perform very well. It resulted in a score of 1.7925 on kaggle. Before submitting to kaggle it is mandatory to handle the negative or null values. One thing needed is probably more time training the model. The model would perform much better with additional time. However, if the time is to be kept constant, we can try adding features and categorizing data which arent actually integers. Then we can fine tune using the hyperparameters. The parameters used here are:
nn_options = {  
    'num_epochs': 10,  
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  
    'layers': ag.space.Categorical([100],[1000],[200,100],[300,200,100]), 
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1), 
}

gbm_options = {  
    'num_boost_round': 100,  
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36), 
}

  
search_strategy = 'auto'
number_of_trials = 3
hyperparameter_tune_kwargs = { 
    'num_trials': number_of_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}

hyperparameters = {'GBM': gbm_options,'NN': nn_options} 


### What was the top ranked model that performed?
The best performing model was found to be WeightedEnsemble_L3 
The first 10 entries appeared as:
|leaderboard|model|score_val|pred_time_val|fit_time|
|--|--|--|--|--|
 |0|           WeightedEnsemble_L3|  -30.083829|      12.224649|  529.127609|   
 |1               |LightGBM_BAG_L2  |-30.573790|10.641389|  429.782441   |
 2             |LightGBMXT_BAG_L2|  -30.587060|      11.553783|  431.848863|  
 3               |CatBoost_BAG_L2|  -31.374845|      10.291574|  464.289287|   
 4        |RandomForestMSE_BAG_L2  |-32.756747|      11.096447|  430.442422   |
 5         |  WeightedEnsemble_L2|  -33.594077 |      9.364439|  353.030635|   
 6           |  LightGBMXT_BAG_L1 | -35.459772|       4.846866  | 60.124768   |
 7            |   LightGBM_BAG_L1 | -35.626974|       2.791491|   45.053618|   
 8               |CatBoost_BAG_L1  |-36.837843       |0.422050 | 226.340144   |
 9          |ExtraTreesMSE_BAG_L1 | -38.059678  |     0.592837 |   6.551266 |  



## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Along with the basic statistical functions like min, max, variance, for exploratory analysis, histogram was plotted along with correlation matrix. Data Wrangler makes it easy to visualise the histogram for each column. The additional feature that could be extracted was the datetime split. DateTime can be split into year and month providing more features to train the model with.


### How much better did your model preform after adding additional features and why do you think that is?
There was a slight improvement in the performance of the model after adding year and month as separate features. This part comprises of feature engineering. Removing the casual and registered columns as they are not significant and addition of year and month as additional features. A model needs enough features so that it does not overfit or underfit. Thus the addition of the feature improves the overall performance of the model.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The hyperparameters tried were:
nn_options = {  
    'num_epochs': 10,  
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  
    'layers': ag.space.Categorical([100],[1000],[200,100],[300,200,100]), 
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1), 
}

gbm_options = {  
    'num_boost_round': 100,  
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36), 
}

  
search_strategy = 'auto'
number_of_trials = 3
hyperparameter_tune_kwargs = { 
    'num_trials': number_of_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}

hyperparameters = {'GBM': gbm_options,'NN': nn_options} 

There was a significant improvement after altering the hyperparameters to the model. These can be better visualized in the diagrams below.

### If you were given more time with this dataset, where do you think you would spend more time?
Given more time, I would try multiple rounds of tuning the hyperparameters. Increasing the training time of the model would also help to improve the performance.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|-|-|-|-53.087452|
|add_features|-|-|-|-30.029135|
|hpo|num_trials|scheduler|searcher|-37.238674|


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.
The score for WeightedEnsemble_L3
![model_test_score.png](model_test_score_highest.png)


## Summary
The model was trained using Autogluon to find the bike sharing demand. The best performing model was Weighted Ensemble L3. With additional features, the performance of the model is enchanced. Fine tuning the hyperparameters can be tedious but affects the performance of the model significantly.
