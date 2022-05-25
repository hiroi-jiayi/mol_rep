#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import random
import glob
from scipy import stats
import pandas as pd
import numpy as np
import os

def rfr_prediction(folder):
    

    files = pd.read_csv('random_sel_files.csv')["file_name"].values
    maccs_df = {}

    
    
    for obj in files:
        maccs_df[obj] = pd.read_csv(folder + "/" + obj)      

    
    all_r_2 = []
    all_mse = []
    all_spearman_pval = []
    all_spearman_rho = []
    all_r_2_med = []
    all_mse_med = []
    all_spearman_med_rho = []
    all_spearman_med_pval = []
    unique_col = []
    
    file_len = []
    names = []
    file_name = [name for name in maccs_df]

    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(maccs_df[file_name[i]].iloc[:, 3:], maccs_df[file_name[i]]['pXC50'], test_size = 0.2)
        rfcv=RandomForestRegressor(n_estimators = 1000)
        r2 = []
        mse = []
        spearman_rho = []
        spearman_pval = []
        cv = model_selection.KFold(n_splits=5) 
        for (train, test), j in zip(cv.split(x_train, y_train), range(5)):
            rfcv.fit(x_train.iloc[train], y_train.iloc[train])
            y_pred = rfcv.predict(x_test)
            r2.append(r2_score(y_test, y_pred))
            mse.append(mean_squared_error(y_test, y_pred))
            rho, pval = stats.spearmanr(y_test, y_pred)
            spearman_rho.append(rho)
            spearman_pval.append(pval)
            
        r_2_mean = np.mean(r2)
        r_2_median = np.median(r2)
        mse_mean = np.mean(mse)
        mse_median = np.median(mse)
        spearman_mean_rho = np.mean(spearman_rho)
        spearman_mean_pval = np.mean(spearman_pval)
        spearman_median_rho = np.median(spearman_rho)
        spearman_median_pval = np.median(spearman_pval)
        
        all_r_2.append(r_2_mean)
        all_r_2_med.append(r_2_median)
        
        all_mse.append(mse_mean)
        all_mse_med.append(mse_median)
        
        all_spearman_rho.append(spearman_mean_rho)
        all_spearman_pval.append(spearman_mean_pval)
       
        all_spearman_med_rho.append(spearman_median_rho)
        all_spearman_med_pval.append(spearman_median_pval)
        
        file_len.append(len(maccs_df[file_name[i]]))
        unique_col.append(len(maccs_df[file_name[i]].columns)-3)
        names.append(file_name[i])

    data = {'File name': names,
            'File size':file_len,
            'Unique columns': unique_col,
            'Mean r^2 value': all_r_2,
            'Median r^2 value':all_r_2_med,
            'Mean MSE':all_mse,
            'Median MSE': all_mse_med,
            'Spearman coef rho':all_spearman_rho,
            'Spearman coef rho med':all_spearman_med_rho,
            'Spearman coef pval':all_spearman_pval,
            'Spearman coef pval med':all_spearman_med_pval}
    
    df = pd.DataFrame(data=data)
    print(df)
    df.to_csv('rdkit2_200_rfr1000_1.csv',index=False)

        

        # print('Mean r^2 value is: '+ str(r_2_mean) + 
        #       " Mean MSE is: "+ str(mse_mean) + 
        #       " Spearman coef is:" +str(spearman_mean)+
        #       " file name and len: " +str(file_name[i])+" "+str(len(morgan_df[file_name[i]])))

    
    return 


rfr_prediction('rdkit2/')

