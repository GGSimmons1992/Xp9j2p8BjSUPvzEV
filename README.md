# Happy Customers

## Background 
This is a simple classification project detecting happy and unhappy customers from a mock dataset. 

Data Description:

Y = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 = my order was delivered on time
X2 = contents of my order was as I expected
X3 = I ordered everything I wanted to order
X4 = I paid a good price for my order
X5 = I am satisfied with my courier
X6 = the app makes ordering easy for me

Using the the dataset provided by Apziva, and the key above, the goal is to reach at least 73% model accuracy, or convince the audicence why my solution is superior. External holds feature input data that will be used for predictions

## Data
Data is not pushed due to gitignore. Raw data from Apziva is in Raw folder. Running src/Data/make_dataset.py will insert transformed train and test data into Processed folder. 
## Models
goodModelsDictionary.json has the names of the pickle files of the models that passed. The two models that passed was logModel.pkl (logistic regression model trained on standard scaled data) and tree.pkl (a decision tree with max_depth of 2 and max_features per split of 2)
## Notebooks
eda.ipynb is an eda exploration of the data set. machineLearning.ipynb was used originally for machine learning experimentation then refactored to generate the script file that would eventually become train_model.py
## src
util.py has 3 methods used for earlier experiments for machineLearning.ipynb, but no longer is used by machineLearning.ipynb or any other script file. __init__.py is a boilerplate that could be used for future models; it's just a placeholder for this repo. Models/make_dataset.py has methods to load train and test sets through loadData() or created train and test sets through makeData(). Running Models/train_model.py trains and evaluates the models and stores the passing models in root/Models folder. Running Models/predict_model will make predictions on a features only data set in Data/External (i.e. running "python predict_model.py mockData" will predict on the input dataset Data/External/mockData.csv and print the predictions).
## Requirements.txt
list of python packages used by this repo.
## Conclusion
Of the different attempted models, the ones that were able to achieve above 73% accuracy were the logModel.pkl (logistic regression model trained on standard scaled data) and  the tree.pkl (a decision tree with max_depth of 2 and max_features per split of 2)
