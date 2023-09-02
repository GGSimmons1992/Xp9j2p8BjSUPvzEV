import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
import json
import scipy.stats as stats
import math
from sklearn.model_selection import train_test_split
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier as rf
import sklearn.model_selection as ms
import sklearn.metrics as sm
import sys
sys.path.insert(0, "../src/")
import util as util

class SimpleRandomForestClassifierParams():
    def __init__(self,params):
        self.n_estimators = params.n_estimators
        self.max_features = params.max_features
        self.max_depth = params.max_depth

class SimpleRandomForestClassifier():
    def __init__(self,simplerandomforestclassifierparams):
        self.model = rf(n_estimators = simplerandomforestclassifierparams.n_estimators,
                        max_features = simplerandomforestclassifierparams.max_features,
                        max_depth = simplerandomforestclassifierparams.max_depth)