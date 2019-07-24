#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_selection import chi2 as chi2
import sklearn
from sklearn import ensemble
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

from sklearn.model_selection import train_test_split


class ColSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,percent=1., feature_selector_type="f_classif"):
        
        self._allowed_featureselectors = ["chi2", "f_classif"]       
       
        assert percent<=1,            "the percent introduced {} is not valid. Please a number in 0<percent<=1"
        
        self.percent = percent
        
        assert feature_selector_type in self._allowed_featureselectors,            "the featureselector introduced {} is not valid. Please use one in {}".format(featureselector, self._allowed_featureselectors)
        self.feature_selector_type = feature_selector_type
    
    def fit(self,X,y):
        n_cols = X.shape[1]
        
        self.n_features_selected = int(self.percent * n_cols)
        
        #import pdb;pdb.set_trace()
        
        if self.feature_selector_type == "chi2":
            self.featureselector = sklearn.feature_selection.SelectKBest(chi2,k= self.n_features_selected)
                
        if self.feature_selector_type == "f_classif":
            self.featureselector = sklearn.feature_selection.SelectKBest(f_classif,k= self.n_features_selected)
    
        self.featureselector.fit(X,y)
        return self

    def transform(self, X):
        return self.featureselector.transform(X)

        
        


# In[ ]:


class GradientBoostingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Feature generator from a gradient boosting
    """
    
    def __init__(self,
                 stack_to_X=True,
                 sparse_feat=True,
                 add_probs=True,
                 criterion='friedman_mse',
                 init=None,
                 learning_rate=0.1,
                 loss='deviance',
                 max_depth=3,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_estimators=50,
                 n_iter_no_change=None,
                 presort='auto',
                 random_state=None,
                 subsample=1.0,
                 tol=0.0001,
                 validation_fraction=0.1,
                 verbose=0,
                 warm_start=False):
        
        # Deciding wheather to append features or simply return generated features
        self.stack_to_X  = stack_to_X
        self.sparse_feat = sparse_feat  
        self.add_probs   = add_probs   

        # GBM hyperparameters
        self.criterion = criterion
        self.init = init
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.presort = presort
        self.random_state = random_state
        self.subsample = subsample
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.warm_start = warm_start
        
        
    def _get_leaves(self, X):
        X_leaves = self.gbm.apply(X)
        n_rows, n_cols, _ = X_leaves.shape
        X_leaves = X_leaves.reshape(n_rows, n_cols)
        
        return X_leaves
    
    def _decode_leaves(self, X):

        if self.sparse_feat:
            #float_eltype = np.float32
            #return scipy.sparse.csr.csr_matrix(self.encoder.transform(X), dtype=float_eltype)
            return scipy.sparse.csr.csr_matrix(self.encoder.transform(X))
        else:
            return self.encoder.transform(X).todense()
        
    
    def fit(self, X, y):
        
        self.gbm = sklearn.ensemble.gradient_boosting.GradientBoostingClassifier(criterion = self.criterion,
                            init = self.init,
                            learning_rate = self.learning_rate,
                            loss = self.loss,
                            max_depth = self.max_depth,
                            max_features = self.max_features,
                            max_leaf_nodes = self.max_leaf_nodes,
                            min_impurity_decrease = self.min_impurity_decrease,
                            min_impurity_split = self.min_impurity_split,
                            min_samples_leaf = self.min_samples_leaf,
                            min_samples_split = self.min_samples_split,
                            min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                            n_estimators = self.n_estimators,
                            n_iter_no_change = self.n_iter_no_change,
                            presort = self.presort,
                            random_state = self.random_state,
                            subsample = self.subsample,
                            tol = self.tol,
                            validation_fraction = self.validation_fraction,
                            verbose = self.verbose,
                            warm_start = self.warm_start)
        
        self.gbm.fit(X,y)
        self.encoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
        X_leaves = self._get_leaves(X)
        self.encoder.fit(X_leaves)
        return self
        
    def transform(self, X):
        """
        Generates leaves features using the fitted self.gbm and saves them in R.

        If 'self.stack_to_X==True' then '.transform' returns the original features with 'R' appended as columns.
        If 'self.stack_to_X==False' then  '.transform' returns only the leaves features from 'R'
        Ìf 'self.sparse_feat==True' then the input matrix from 'X' is cast as a sparse matrix as well as the 'R' matrix.
        """
        R = self._decode_leaves(self._get_leaves(X))
        
        if self.sparse_feat:
            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R, scipy.sparse.csr.csr_matrix(P))) if self.stack_to_X==True else R
            else:
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R)) if self.stack_to_X==True else R

        else:

            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R, scipy.sparse.csr.csr_matrix(P))) if self.stack_to_X==True else R
            else:
                X_new =  np.hstack((X, R)) if self.stack_to_X==True else R

        return X_new

