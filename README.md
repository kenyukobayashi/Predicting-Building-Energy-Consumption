# Machine Learning Project : Predicting Energy Building Consumption


## Libraries required
* Numpy
* Pandas
* Pytorch
* Scikit-learn


## Data files

* #### Buildings' dataset : *sanitized_complete.csv*
This dataset contains every buildings defined by their associated heating/cooling energy demands and their features :
* EGID : The building id.
* heating : The yearly energy demand for heating, in Wh.
* cooling : The yearly energy demand for cooling, in Wh.
* GASTW : The number of floors of the building.
* GAREA : The surface of the building, in m2.
* habit : Boolean which specifies whether the building is a habitation or not.
* b19 : Boolean which specifies whether the building was built before 1919 or not.
* xx-yy : Boolean which specifies whether the building was built between 19xx and 19yy or not.
* a2000 : Boolean which specifies whether the building was built after 2000 or not.

#### * Forecasts' dataset : *daily_forecast.csv*
This dataset contains daily weather forecasts defined by the day and their different features :
* Timestamp : The day in the yyyy-mm-dd format.
* h_day : The hour of the day
* G_Dh_min/mean/var :
* G_Bn_min/mean/var :
* Ta_min/mean/var : The min/mean/var of the temperature, in degree celsius.
* FF_min/mean/var :
* DD_min/mean/var :

#### * Daily predictions data : *daily_predictions.csv*
The daily predictions made by our ANN model :
* Timestamp : The day in the *yyyy-mm-dd* format.
* xxxxxxx : The EGID of the building (building id), for which its heating demands are given for each timestamp.


## Running

The run.py file does the whole job :<br/>
Usage: `python3 run.py [regression|svr|ann|daily]`<br/>
Runs the models with 4-fold cross validation:<br/>
* for 'regression', 'svr' or 'ann', runs the annual prediction for the regression (least square, baseline), svr or the neural network.
* for 'daily', runs the neural network for daily prediction.<br/>
Note that the ANNs can take a long time to run, so their progresses are printed during learning.<br/>
At the end, prints the average Ln Q error on testing and the standard deviation.


## Python files

#### * preprocessor.py :
*Class* : **DataPreProcessor**<br/>
Given a dataset in csv format, it allows to :
* Split data :
Specify the ratio with the *split* parameter in the constructor.<br/>
The *.train/test* attribute returns a Panda Dataframe instance of the data,
while the *.train/test_loader* attribute returns a DataLoader instance of the data.
* Normalize data :
Specify the columns to normalize with the *columns_to_normalize* parameter in the constructor.<br/>
Specify the use of the logarithm for the normalization with *use_log* parameter in the constructor.
* Evaluate data :
Given a dataset and associated predictions made by a model, the *evaluate* method computes different losses of the predictions.

#### * cross_validation.py :
*Class* : **CrossValidation**<br/>
Given a dataset in a csv format, it returns an array of k DataPreProcessor instances,
allowing a K-fold Cross Validation process to be done on the dataset :
* Split ratio :
Specify the ratio with the *k* parameter in the constructor.

#### * correlations.py :
Computes the pair wise correlations on the building features and the weather forecast features.
Presents the results in the form of heatmaps.

#### regression.py :
Implements Ridge Regression and Least Squares method.

#### * svr.py :
*Class* : **Svr**<br/>
Implements Support Vector Regression method, with the following parameters to be specified in the constructor :
* data : DataPreProcessor instance of the dataset : DataPreProcessor
* kernel_type : Specifies the kernel type to be used : string
* gamma : Kernel coefficient. Use if kernel_type = ‘rbf’, ‘poly’ or ‘sigmoid’ : string/float
* degree : Degree of the polynomial kernel function. Use if kernel_type = ‘poly’ : int
* coef0 : Independent term in kernel function. Use if kernel_type = ‘poly’ or ‘sigmoid’ : float
* epsilon : Epsilon value in the epsilon-SVR model : float
* c : Regularization parameter : float
* Shrinking : Specifies whether to use the shrinking heuristic or not : boolean
* tolerance : Tolerance for stopping criterion : float
Note that these hyper parameters have default values to be assigned if not specified.
The best hyper parameters given by the grid-search algorithm are specified in the run_training function.

#### * ann.py :
*Class* : **Ann**<br/>
Implements Artificial Neural Networks method, with the following parameters to be specified in the constructor :
* data :  DataPreProcessor instance of the dataset, DataPreProcessor
* n_features : Number of nodes on a layer : int
* n_output : Number of output nodes : int
* n_hidden : Number of hidden layers : int