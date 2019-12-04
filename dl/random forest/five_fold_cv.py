import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt


def cross_validation_gbdt(x, y, train_test_split, gbm):

    print('nice')