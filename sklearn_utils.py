

import copy
import multiprocessing as mp
import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

def fit_cv_subsample (pipe_cv, X, y, n_max = 10_000):
    '''
    This function fits a CV in a subsample of the first n_max rows
    returns the trained pipe and the best estimator
    '''
    X_sub = X[0:n_max]
    y_sub = y[0:n_max]
    pipe_cv.fit(X_sub,y_sub)
    #pipe_cv.best_estimator_.fit(X,y)
    return pipe_cv, pipe_cv.best_estimator_


def fit_and_return_preds(p_id,
                         return_predictions,
                         model, 
                         X:                   pd.core.frame.DataFrame or np.ndarray,
                         y:                   pd.core.frame.DataFrame or np.ndarray,
                         tr_idx:              np.ndarray or list,
                         va_idx:              np.ndarray or list, 
                         X_te:                pd.core.frame.DataFrame or np.ndarray,
                         evaluation_metric):
    """
    This function is meant to be used to train a model on the rows of `X` specified by `tr_idx`.
    Then it computes valudation metric in the rows `val_idx`.
    Finally it computes predictions on `X_te`
    """
    assert type(X) in [pd.DataFrame, np.ndarray], "type(X)={} but it should be a pd.DataFrame or np.ndarray".format(type(X))
    assert type(y) in [pd.DataFrame, np.ndarray], "type(y)={} but it should be a pd.DataFrame or np.ndarray".format(type(X))
    assert type(tr_idx)== np.ndarray, "type(tr_idx)={} but it should be a np.ndarray".format(type(train_idx))
    assert type(va_idx)== np.ndarray, "type(va_idx)={} but it should be a np.ndarray".format(type(train_idx))
    
    
    if type(X) == pd.DataFrame:
        X_tr, X_va = X.iloc[tr_idx, :],  X.iloc[va_idx, :]
        y_tr, y_va = y.iloc[tr_idx],     y.iloc[va_idx]
    else:
        X_tr, X_va = X[tr_idx, :],  X[va_idx, :]
        y_tr, y_va = y[tr_idx],     y[va_idx]
        
    model    = model.fit(X_tr, y_tr)
    y_te_hat = model.predict(X_te)
        
    y_tr_pred = model.predict_proba(X_tr)[:,1]
    y_va_pred = model.predict_proba(X_va)[:,1]
    print('{} train: {}'.format(evaluation_metric.__name__, evaluation_metric(y_tr, y_tr_pred)))
    print('{} valid: {}'.format(evaluation_metric.__name__, evaluation_metric(y_va, y_va_pred)))

    y_te_pred = model.predict_proba(X_te)[:,1]
    
    return_predictions[p_id] = y_te_pred




def fit_and_average_k_models(model, X, y, X_te, k):
    
    manager            = mp.Manager()
    return_predictions = manager.dict()
    evaluation_function = sklearn.metrics.roc_auc_score
    kf = sklearn.model_selection.KFold(n_splits = k, shuffle = True)
    
    processes=[]
    for p_id, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        processes.append(mp.Process(target=fit_and_return_preds, 
                                    args=(p_id, return_predictions, 
                                          copy.deepcopy(model),
                                          X, y , tr_idx, va_idx, X_te, evaluation_function)))

    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
        
    y_pred = np.zeros(X_te.shape[0])
    for y_k in return_predictions.values():
        y_pred += y_k
    
    y_pred = y_pred/k
    
    return y_pred, return_predictions

