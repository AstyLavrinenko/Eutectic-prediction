''' '''
import pandas as pd
import numpy as np
from time import time

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def split_by_bins(df, test_size):
    df_bins = [b for _, b in df.groupby(by = ['bins'])]
    train_idx = []
    test_idx = []
    for data in df_bins:
        data_train, data_test = train_test_split(data, test_size = test_size, random_state = 42)
        train_idx += list(data_train.index)
        test_idx += list(data_test.index)
    return train_idx, test_idx
    
def evaluate_model(model, x, y, scaler_y=False):
    y_pred = model.predict(x)
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))    
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared = False)
    
    return r2, mae, rmse

def fit_model_kfold(ML_model, ML_name, x_train, y_train, scaler_y=False, scaler_x=StandardScaler(), n_kfold=10): 
    cv_results = pd.DataFrame()
    kfold = KFold(n_splits = n_kfold, shuffle = True, random_state = 42)
    for train_idx, val_idx in kfold.split(x_train):
        model = ML_model        
        
        x_train_new, x_val = x_train[train_idx], x_train[val_idx]
        y_train_new, y_val = y_train[train_idx], y_train[val_idx]        
        scaler_x = scaler_x
        x_train_new = scaler_x.fit_transform(x_train_new)
        x_val = scaler_x.transform(x_val)
        if scaler_y:
            scaler_y = StandardScaler()        
            y_train_new = np.ravel(scaler_y.fit_transform(y_train_new))
        
        init_time = time()
        model.fit(x_train_new, y_train_new)
        fit_time = time() - init_time
        
        if scaler_y:
            y_train_new = scaler_y.inverse_transform(y_train_new.reshape(-1, 1))        
        r2_train, mae_train, rmse_train = evaluate_model(model, x_train_new, y_train_new, scaler_y)
        r2_val, mae_val, rmse_val = evaluate_model(model, x_val, y_val, scaler_y)        
        
        cv = {
        'fit_time': fit_time,
        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'r2_val': r2_val,
        'mae_val': mae_val,
        'rmse_val': rmse_val}
        cv_results = cv_results.append(cv, ignore_index = True)        
    
    cv_results = cv_results.agg(['mean', 'std'])
    result_dict = {'model_name': ML_name, 'model_params': model.get_params()}
    for col in cv_results.columns:
        for idx in cv_results.index:
            result_dict[f'{col}_{idx}'] = cv_results.loc[idx, col]           
    
    return result_dict