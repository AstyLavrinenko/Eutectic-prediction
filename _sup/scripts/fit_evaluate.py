''' '''
import pandas as pd
import numpy as np
from time import time

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def custom_cv(x,y,groups,n_splits,test_size):
    custom_cv = []
    kfold = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    for train_idx, test_idx in kfold.split(x, y, groups):
        custom_cv.append((train_idx, test_idx))
    return custom_cv    

def evaluate_model(model, x, y, scaler_y=False,results=False, ML_name=None):
    names = {
        'RFR': 'Random Forest Regression', 
        'GBR': 'Gradient Boosting Regression', 
        'KNN': 'k-Nearest Neighbors Regression', 
        'SVR': 'Support Vector Regression', 
        'MLR': 'Multiple Linear Regression'
        }
    y_pred = model.predict(x)
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))    
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    mape = 100*mean_absolute_percentage_error(y, y_pred)
    if results:
        plt.figure(figsize=(6,6))
        plt.plot(y,y,color='navy', linewidth = 2)
        plt.scatter(y_pred,y,s=50,color='maroon')
        plt.title(f'{names[ML_name]}', {'fontsize': 14})
        plt.xlabel('Predicted temperature, K', {'fontsize': 14})
        plt.xticks([250,350,450,550],fontsize = 14)
        plt.ylabel('Experimental temperature, K', {'fontsize': 14})
        plt.yticks([250,350,450,550],fontsize = 14)
        plt.savefig(f'../results/{ML_name}.svg',dpi=1200)
        plt.show()
    return r2, mae, rmse, mape

def fit_model_kfold(ML_model, ML_name, x_train, groups, y_train, scaler_y=False, scaler_x=MinMaxScaler(), n_kfold=5, validation=False,y_val=None,x_val=None): 
    cv_results = pd.DataFrame()
    kfold =  custom_cv(x_train,y_train,groups,n_kfold,1/n_kfold)
    k=-1
    importance = []
    for train_idx, test_idx in kfold:
        k+=1
        model = ML_model        
        
        x_train_new, x_test = x_train[train_idx], x_train[test_idx]
        y_train_new, y_test = y_train[train_idx], y_train[test_idx]        
        scaler_x = scaler_x
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_test = scaler_x.transform(x_test)
        if scaler_y:
            scaler_y = StandardScaler()        
            y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
        init_time = time()
        model.fit(x_train_new, np.ravel(y_train_new))
        fit_time = time() - init_time
        
        importance.append(permutation_importance(model, x_train_new, np.ravel(y_train_new), scoring='neg_mean_squared_error').importances_mean)
        
        if scaler_y:
            y_train_new = scaler_y.inverse_transform(y_train_new.reshape(-1, 1))        
        r2_train, mae_train, rmse_train, mape_train = evaluate_model(model, x_train_new, y_train_new, scaler_y)
        r2_test, mae_test, rmse_test, mape_test = evaluate_model(model, x_test, y_test, scaler_y)        
        
        cv = {
        'fit_time': fit_time,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'mape_train': mape_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'mape_test': mape_test}
        
        for key,value in cv.items():
            cv_results.loc[k,key] = value
    cv_results = cv_results.agg(['mean', 'std'])
    result_dict = {'model_name': ML_name, 'model_params': model.get_params()}
    for col in cv_results.columns:
        for idx in cv_results.index:
            result_dict[f'{col}_{idx}'] = cv_results.loc[idx, col]   
    if validation:
        model = ML_model
        scaler_x = scaler_x
        x_train_new = scaler_x.fit_transform(x_train)
        x_val_new = scaler_x.transform(x_val)
        y_train_new = y_train
        y_val_new = y_val
        if scaler_y:
            scaler_y = StandardScaler()        
            y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        model.fit(x_train_new, np.ravel(y_train_new))
        r2_val, mae_val, rmse_val, mape_val = evaluate_model(model, x_val_new, y_val_new, scaler_y,results=True,ML_name=ML_name)
        for key,value in {'r2_val':r2_val,'mae_val':mae_val,'rmse_val':rmse_val,'mape_val':mape_val}.items():
            result_dict[f'{key}'] = value
        
    return result_dict, importance