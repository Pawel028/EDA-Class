import pandas as pd
import numpy as np
from EDA import EDA_obj

data = pd.read_csv('train.csv')
data_EDA = EDA_obj(data,'Exited')
data_EDA.create_corr_heat_map()