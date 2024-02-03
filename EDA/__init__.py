from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class EDA_obj:
    def __init__(self,data,depvar,indep_vars=[]):
        self.data = data
        self.depvar = depvar
        self.tracker = [""]*len(data[depvar])
        self.usable_vars = indep_vars

    def find_numercial_vars(self):
        return self.data.columns[self.data.dtypes == float]
    
    def find_categorical_vars(self):
        return self.data.columns[self.data.dtypes != float]
    
    def create_corr_heat_map(self):
        numerical_vars = self.data[self.find_numercial_vars()]
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_vars.corr(), annot=True, cmap="OrRd", fmt='.3f', cbar=True, vmin=-1, vmax=1)
        plt.show()
