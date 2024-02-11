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
    
    def univariate(self):
        varlist_num = self.find_numercial_vars()
        varlist_cat = self.find_categorical_vars()

        f, ax = plt.subplots(1, 1, figsize=(10, 5))
        for var in varlist_cat:
            sns.countplot(x=var, data=self.data, hue=var, palette=None, order=None)
            plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
            ax.set_title(f"Frequencies of {var}")
            plt.show()

        # for var in varlist_num:
        l=len(varlist_num)
        fig = plt.figure(figsize =(10, 5)) 
        n = 0
        for var in varlist_num:
            print(n,var)
            ax = fig.add_axes([n/l+0.02,0.03,1/l,0.9])
            # Creating plot
            plt.boxplot(self.data[var])
            ax.set_title(f"Box Plot of {var}")
            n=n+1
        plt.show()
        return 0
    
    def variablility(self,var): # Variablity in categorical data
        if self.data[var].dtypes == object:
            data = pd.DataFrame(self.data[var], columns = [var])
            data['ones'] = 1
            data1 = data.groupby(['SMOKE']).count()
            data1=data1/data1['ones'].sum()
            x=(data1*data1).sum()
            return 1-x['ones']
        else:
            return "Please Enter a Categorical Variable"

    def IQR(self,var): # for any numerical Data
        if self.data[var].dtypes != float:
            return "Please Choose a Numerical Variable"
        else:
            x=self.data[var]
            q25 = x.quantile(.25)
            q75 = x.quantile(.75)
            return [q25-1.5*(q75-q25), q75-1.5*(q75-q25)]
        
    def euclidean_distance_outlier(self,varlist, cutoff):
        # x=np.array(self.data[[var1, var2]])
        x=np.zeros([len(self.data), len(varlist)])
        i=0
        for var in varlist:
            x[:,i] = self.data[var]
            i=i+1

        data_mean = x.mean(axis=0) # mean of data
        data_1 = ((x-data_mean)**2).sum(axis=1) # distance of each point from the mean
        result = pd.Series([0] * len(x))
        lb = data_1.mean()-cutoff*data_1.std()
        ub = data_1.mean()+cutoff*data_1.std()
        result[(data_1<lb)] = 1
        result[(data_1>ub)] = 1
        return result.sum()


    def bivariate():
        return 0
    
    def detect_outlier():
        return 0
    
    def dimensionality_reduction():
        return 0
    
    def standardize_data():
        return 0
    




