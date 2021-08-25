# Understanding-of-Optuna-A-Machine-Learning-Hyperparameter

![image](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/intro.png)

### Preface
This article aims to provide consolidated information on the underlying topic of “Optuna” and is not to be considered as the original work. The information and the code are repurposed through several online articles, research papers, books, and open-source code. The article aims to explain what Optuna is and how we can use to obtain optimal values of model hyperparameters. We will explain all of the required concepts in simple terms along with outlining how we can implement the Optuna in Python.

### Introduction
Hyperparameter Tuning is searching for the right set of parameters that can help to achieve high precision and accuracy of a model. Optimising hyperparameters constitute one of the trickiest parts in building the machine learning models. It is nearly impossible to predict the optimal parameters while building a model, at least in the first few attempts. The primary aim of hyperparameter tuning is to find the sweet spot for the model’s parameters so that a better performance is obtained.

 ![image](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/hyperparameter%20tunning.png)
 
 fig. Hyperparameter Tunner 
 
In the figure above, it can be seen that the “hyperparameter tuner” is external to the model and the tuning is done before model training. The result of the tuning process is the optimal values of hyperparameters which is then fed to the model training stage. 

Hyperparameter optimization is one of the crucial steps in training Machine Learning models. With many parameters to optimize, long training time and multiple folds to limit information leak, it may be a cumbersome endeavour. 

There are a few methods of dealing with the issue, the two most widely used parameter optimization techniques: 

•	Grid search - Grid-search is used to find the optimal hyperparameters of a model which results in the most ‘accurate’ predictions. Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters. It is an exhaustive search that is performed on the specific parameter values of a model. The model is also known as an estimator. Grid search exercise can save us time, effort and resources.

•	Random search – Random: Randomly samples the search space and continues until the stopping criteria are met.  In contrast to GridSearchCV, in case of the random search, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter.

Then why use Optuna? Following features are Optuna answers that,

•	Eager dynamic search spaces

•	Efficient sampling and pruning algorithms

•	Easy integration

•	Good visualizations

•	Distributed optimization

Let me now introduce Optuna, an optimization library in Python that can be employed for hyperparameter optimization.



### Tuning Hyperparameters with Optuna
Optuna was developed by the Japanese AI company [Preferred Networks](https://www.preferred.jp/en/), is an open-source automatic hyperparameter optimization framework, automates the trial-and-error process of optimizing the hyperparameters. It automatically finds optimal hyperparameter values based on an optimization target. Optuna is framework agnostic and can be used with most Python frameworks, including keras, Scikit-learn, Pytorch, etc. This framework is mainly designed for machine learning and can be used on non-ML task until unless we can define objective function. Optuna features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.


### Optuna Implementation 
Link for the complete codes implemented in this article can be found [here](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optuna_Introduction_Non_ML_Task_Demo.ipynb) .
Optuna architecture is capable of handling both small- and large-scale experiments with minimum setup requirements. Optuna installation in python via pip:

```!pip install optuna```


Basic Structure of code implementation with optuna:
 
In the image above, import the optuna package, create an objective function with parameter trial (specifies the number of trials), write your machine learning model within the function and return the trained model’s evaluation. Now, create an optuna study and specify that particular objective function(minimize/maximize). Finally, optimize your created study by passing the objective function and number of trials.

•	Trial: A single call of the objective function

•	Study: An optimization session, which is a set of trials

•	Parameter: A variable whose value is to be optimized,

 
### Example 1: Non-ML Task 

Here ‘ll try to demo non-ML task to describe how to use optuna step-by-step.Code Snippet is available [here](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optuna_Introduction_Non_ML_Task_Demo.ipynb).

![image](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optimization%20process%20overview%20for%20Non-ML%20task.jpg) 
 
Fig. Optimization Process overview for Non-ML task 

* Create Objective Function :

Functions that are to be optimized are named Objective. Our goal is to find the value of x that minimizes the output of the objective function. This is the “optimization.” During the optimization, Optuna repeatedly calls and evaluates the objective function with different values of x.

```
def objective(trial):
    #trial suggesting the values for x between -100 to 100
    x = trial.suggest_uniform('x', -100, 100)
    #return the funciton value
    return (x - 2) ** 2
```

The suggest APIs (for example, suggest_float()) are called inside the objective function to obtain parameters for a trial. suggest_float() selects parameters uniformly within the range provided. In our example, from −10 to 10.

* We can dynamically construct Pythonic Search Space: 

For hyperparameter sampling, Optuna provides the following features:

optuna.trial.Trial.suggest_categorical() for categorical parameters

optuna.trial.Trial.suggest_int() for integer parameters

optuna.trial.Trial.suggest_float() for floating point parameters


We can check more details [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py) .

* Create Study Object and Optimize it: 

In Optuna, we use the study object to manage optimization. Method create_study() returns a study object. A study object has useful properties for analysing the optimization outcome.  To start the optimization, we create a study object and pass the objective function to method optimize() as follows,

```
study = optuna.create_study(direction='minimize') #Set minimize for minimization and maximize for maximization.
#To start the optimization, we create a study object and pass the objective function to method
study.optimize(objective, n_trials=100)
```



* Get Best Parameters:

######  * To get the dictionary of parameter name and parameter values:
```print("Return a dictionary of parameter name and parameter values:",study.best_params)```

######  * To get the best trial:
```print("\nReturn the best trial:",study.best_trial)```

######  * To get all trials:
```print("\nReturn all the trials:", study.trials)```

######  * To get the number of trials:
```print("\nReturn the number of trials :",len(study.trials))```

### Example 2:  Optimize Machine Learning Classifier Model - RandomForest Classifier
 
 ![image](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optimization%20Process%20overview%20for%20ML%20task.jpg)
 
Fig. Optimization Process Overview for ML task 

1.	Import all the required packages. The code snippet is available [here](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optuna_Example_on_Heartdisease_Dataset_for_Randomforest_Classifier.ipynb).
2. Define an objective function with the trial parameter. The objective function should contain the machine learning logic i.e., fit the model on data(heart disease  dataset), predict the test data and return the evaluation score as illustrated below.


```
def objective(trial):
    
    rf_max_depth = trial.suggest_int("rf_max_depth", 1, 7, log=True)
    rf_max_features = trial.suggest_int("rf_max_features",1,10,log=True)
    rf_estimators = trial.suggest_int('rf_estimators', 10,100,step=10)
    classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_estimators,max_features = rf_max_features)

    score = cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=5)
    accuracy = score.mean()
    return accuracy
```

Here we are trying to optimize 3 parameters max_depth , max_features and extimators.
Firstly, we defined search space for these hyperparameters. As seen, n_estimators are integer ranging from 10 to 100 with step of 10, max_depth is taken from log uniform ranging from 1 to 7 and max_features is taken from log uniform withing range of 1 to 10 .Values and type of values can be changed depending on type of parameter, e.g. we can use suggest_categorical to specify the classifiers.

Secondly, evaluate the objective function value using study object. Direction can be ‘maximize’ or ‘minimize’ depending upon the nature of the objective. Here we need to maximize the cross_val_score, other parameter to optimize func is No of trials as 100.
We haven’t specified the sampler to be used here, by default it is the bayesian optimizer. Once we call the optimize method, the optimization process starts.

```
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print("\n\nBEST PARAMETERS : \n",study.best_params)#get best parameters
```

* Visualization Plots available in Optuna :
![image](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/intro.png)


The visualization module provides utility functions for plotting the optimization process using plotly and matplotlib. Plotting functions generally take a study object and optional parameters are passed as a list to the params argument.
Check [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py) for more detail on visualization for hyperparameter optimization analysis 

### Efficient Sampling and Pruning 

The cost-effectiveness of hyperparameter optimization framework is determined by the efficiency of 

(1) searching strategy that determines the set of parameters that shall be investigated, and 

(2) performance estimation strategy that estimates the value of currently investigated parameters from learning curves and determines the set of parameters that shall be discarded.

**Note:** The strategy for the termination of unpromising trials is often referred to as pruning in many literatures, and it is also well known as automated early stopping. The strategy for the termination of unpromising trials is often referred to as pruning in many literatures, and it is also well known as automated early stopping.

#### •	Sampling Methods on Dynamically constructed Parameter Space 

There are two types of sampling method:

1.	Relational sampling that exploits the correlations among the parameters and 
2.	Independent sampling that samples each parameter independently.

Being an open-source software, Optuna also allows the user to use his/her own customized sampling procedure.
Check [here](https://optuna.readthedocs.io/en/stable/reference/samplers.html) for more details on various sampling methods. 

#### •	Efficient Pruning Algorithm
Pruning mechanism in general works in two phases. 
1. periodically monitors the intermediate objective values, and 
2. terminates the trial that does not meet the predefined condition.

In Optuna, ‘report API’ is responsible for the monitoring functionality, and ‘should prune API’ is responsible for the premature termination of the unpromising trials The background algorithm of ‘should prune’ method is implemented by the family of pruner classes. Check [here](https://optuna.readthedocs.io/en/stable/reference/pruners.html) for more details on pruners.

### Example 3: Implementation of Pruning algorithm with Optuna

Here is the Optuna example that demonstrates a pruner for Keras. In this example, we optimize the validation accuracy of hand-written digit recognition using Keras and MNIST, where the architecture of the neural network and the learning rate of optimizer is optimized. Throughout the training of neural networks, a pruner observes intermediate results and stops unpromising trials. Code Snippet is available [here](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optuna_Demo_with_Keras_DL_Network.ipynb) 

```
def create_model(trial):
    # We optimize the number of layers, hidden units and dropout in each layer and
    # the learning rate of RMSProp optimizer.

    # We define our Sequential Model.
    n_layers = trial.suggest_int("n_layers", 1, 5)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(Dense(num_hidden, activation="relu"))
        dropout = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        model.add(Dropout(rate=dropout))
    model.add(Dense(CLASSES, activation="softmax"))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.RMSprop(learning_rate=lr),
        metrics=["accuracy"],
    )

    return model
```
```
# The KerasPruningCallback checks for pruning condition every epoch.
    model.fit(
        x_train,
        y_train,
        batch_size=BATCHSIZE,
        callbacks=[KerasPruningCallback(trial, "val_accuracy")],
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        verbose=1,
    )
```
```
#Optimize
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=15)
```

 


### Summary

Optuna is the first-choice optimization framework. It’s easy to use, makes it possible to set the study’s timeout, continue the study after a break and access the data easily. Optuna can be used in Machine learning Projects with good results. Preferred Network achieved second place in the [Google AI Open Images 2018 – Object Detection Track](https://www.preferred.jp/en/news/pr20180907) competition.

Optuna is vast software framework. This article demonstrates the few simple examples how optuna can be used. You can explore more examples and optuna reference API through their official website and other blogs present on internet.




### Reference 

1.	[GitHub](https://github.com/KalyaniAvhale/Understanding-of-Optuna-A-Machine-Learning-Hyperparameter/blob/main/Optuna_Introduction_Non_ML_Task_Demo.ipynb) for Code Snippet implemented in this article 
2.	[Optuna Official Website](https://optuna.readthedocs.io/en/stable/index.html) 
3.	[sklearn](https://scikit-learn.org/stable/index.html)
4.	Optuna: A Next-generation Hyperparameter Optimization Framework – [Paper](https://arxiv.org/abs/1907.10902) 
5.	[Key Features Guide](https://optuna.readthedocs.io/en/stable/tutorial/index.html#key-features)
6.	[Optuna Official video guide](https://youtu.be/P6NwZVl8ttc) 


##### Authors : [Kalyani Avhale](https://github.com/KalyaniAvhale) and Siddhi Jadhav

**Your feedback about the article is more than welcomed! Please let us know what you think!**

