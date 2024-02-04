from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

    def map_categorical_var(self,var,dict):
        data1 = self.data.copy(deep = True)
        data1[var] = self.data[var].map(dict)
        self.data = data1

    def chi_squared_test(self, var1, var2):
        data_crosstab = pd.crosstab(self.data[var1], self.data[var2],margins = False) 
        return chi2_contingency(data_crosstab)
    
    def chi_squared_mat(self, heatmap_ind):
        x,y=0,0
        varlist = self.find_categorical_vars()
        mat = np.zeros((len(varlist),len(varlist)))
        # print(varlist)
        for var1 in varlist:
            y=0
            for var2 in varlist:
                # print(x,y)
                pval = chi2_contingency(pd.crosstab(self.data[var1], self.data[var2],margins = False)).pvalue
                mat[y][x] = pval
                y+=1                
            x+=1

        mat1=pd.DataFrame(mat, columns = varlist, index = varlist)
        if heatmap_ind:
            plt.figure(figsize=(10, 5))
            sns.heatmap(mat1, annot=True, cmap="OrRd", fmt='.3f', cbar=True, vmin=-1, vmax=1)
            plt.show()
        return mat1



