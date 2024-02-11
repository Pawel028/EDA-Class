import pandas as pd
import numpy as np
from EDA import EDA_obj

data = pd.read_csv('train.csv')
data = data.drop('id', axis=1)
data_EDA = EDA_obj(data,'NObeyesdad')
<<<<<<< Updated upstream
a=2

data_EDA.map_categorical_var('Gender', {'Male':0, 'Female':1})
print(data_EDA.chi_squared_test('Gender','family_history_with_overweight'))
print(data_EDA.chi_squared_mat(True))
print(data_EDA.variablility('SMOKE'))
=======
# data_EDA.map_categorical_var('Gender', {'Male':0, 'Female':1})
# print(data_EDA.chi_squared_test('Gender','family_history_with_overweight'))
# print(data_EDA.chi_squared_mat(True))
print(data_EDA.euclidean_distance_outlier(['Age', 'Weight'], 3))
>>>>>>> Stashed changes
