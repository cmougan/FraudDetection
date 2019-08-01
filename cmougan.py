
# coding: utf-8

# Carlos Mougan
import pandas as pd
import numpy as np
import scipy
import sklearn
import category_encoders
from sklearn.base import BaseEstimator, TransformerMixin

def missing_data(data):
    '''
    Receives a dataframe and return the type of the column, the total NaNs of the column and the percentage they represent
    '''
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

def resumetable(df):
    '''
    Provides a short summary of a given dataframe
    '''
    
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(scipy.stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary.sort_values('Entropy',ascending=False)

def contains_nan(df_col):
    '''
    This functions checks if a certain column has nans
    '''
    return df_col.isna().any()

def create_cols_for_cols_with_nans(df, inplace=False):
    '''
    This function applied to a dataframe returns a list with the columns with NaNs
    and also returns a data frame with a column with 1 if value is NaN else 0 for all the columns with nans.
    '''
    if inplace:
        cols_with_nan = []
        for c in df.columns:
            if contains_nan(df[c]):
                cols_with_nan.append(c)
                df[c + "_nan"] = df[c].isna().values        
        return cols_with_nan
    else:
        df_copy = df.copy(deep=True)
        cols_with_nan = []
        for c in df.columns:
            if contains_nan(df[c]):
                cols_with_nan.append(c)
                df_copy[c + "_nan"] = df[c].isna().values
        return cols_with_nan, df_copy


def create_statististical_columns_for_nans(df,do_mean=True,do_median=True,do_mode=True,
                                           do_skew=True,do_kurtosis=True,do_std=True):
    '''
    This function applied to a dataframe returns  a data frame with a column with different statistical values
    if value is NaN else 0 for all the columns with nans
    '''
    
    df_copy = df.copy(deep=True)
    cols_with_nan = []
    for c in df.columns:
        if contains_nan(df[c]):
            if do_mean and df[c].dtype !='object':
                media = df[c].mean()
                df_copy[c+'_nan_mean'] = df[c].apply(lambda x: media if np.isnan(x) else 0)
            if do_median and df[c].dtype !='object':
                mediana = df[c].median()
                df_copy[c+'_nan_median'] = df[c].apply(lambda x: mediana if np.isnan(x) else 0)
            if do_mode:
                moda = df[c].mode()
                print(c)
                print(moda)
                #import pdb;pdb.set_trace()
                #df_copy[c + '_nan_mode'] = df[c].apply(lambda x: moda[0] if np.isnan(x) else 0)
            
            if do_std and df[c].dtype !='object':
                deviation = df[c].std()
                df_copy[c+'_nan_std'] = df[c].apply(lambda x: deviation if np.isnan(x) else 0)
            
            
            if do_skew and df[c].dtype !='object':
                skew = df[c].skew()
                df_copy[c+'_nan_skew'] = df[c].apply(lambda x: skew if np.isnan(x) else 0)
                
            if do_kurtosis and df[c].dtype !='object':
                kurtosis = scipy.stats.kurtosis(df[c].dropna())
                df_copy[c+'_nan_kurtosis'] = df[c].apply(lambda x: kurtosis if np.isnan(x) else 0)
                
                
    return df_copy
class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Transformer that filters a type of columns of a given data frame.
    '''
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        #print("Type Selector out shape {}".format(X.select_dtypes(include=[self.dtype]).shape))
        #print(X.select_dtypes(include=[self.dtype]).dtypes)
        return X.select_dtypes(include=[self.dtype])


class Encodings(BaseEstimator, TransformerMixin):
    '''
    This class implements fit and transform methods that allows to encode categorical features in different ways.
    
    '''
    
    def __init__(self, encoding_type="TargetEncoder",columns="All",return_categorical=True):
        #cols: list -> a list of columns to encode, if All, all string columns will be encoded.
        
        self._allowed_encodings = ["TargetEncoder","WOEEncoder","CatBoostEncoder","OneHotEncoder"]           
        assert encoding_type in self._allowed_encodings, "the encoding type introduced {} is not valid. Please use one in {}".format(encoding_type, self._allowed_encodings)
        self.encoding_type = encoding_type
        
        self.columns = columns
        self.return_categorical = return_categorical
        
        
    def fit(self,X,y):
        """
        This method learns encodings for categorical variables/values.
        """
        
        #import pdb;pdb.set_trace()
        
        # Obtain a list of categorical variables
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
    
        # Split the data into categorical and numerical
        self.data_encode = X[self.categorical_cols]

        
        # Select the type of encoder
        if self.encoding_type == "TargetEncoder":
            self.enc = category_encoders.target_encoder.TargetEncoder()
            
        if self.encoding_type == "WOEEncoder":
            self.enc = category_encoders.woe.WOEEncoder()
            
        if self.encoding_type == "CatBoostEncoder":
            #This is very similar to leave-one-out encoding, 
            #but calculates the values “on-the-fly”.
            #Consequently, the values naturally vary during the training phase and it is not necessary to add random noise.
            # Needs to be randomly permuted
            # Random permutation
            perm = np.random.permutation(len(X))
            self.data_encode = self.data_encode.iloc[perm].reset_index(drop=True)
            y = y.iloc[perm].reset_index(drop=True)
            self.enc = category_encoders.cat_boost.CatBoostEncoder()
            
        if self.encoding_type == "OneHotEncoder":
            self.enc = category_encoders.one_hot.OneHotEncoder()
            
            # Check if all columns have certain number of elements bf OHE
            self.new_list=[]
            for col in self.data_encode.columns:
                if len(self.data_encode[col].unique())<50:
                    self.new_list.append(col)
                    
            self.data_encode = self.data_encode[self.new_list]
        
        # Fit the encoder
        self.enc.fit(self.data_encode,y)
        return self

    def transform(self, X):
        
        
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
       
    
        # Split the data into categorical and numerical
        
        self.data_encode = X[self.categorical_cols]
        
        # Transform the data
        self.transformed = self.enc.transform(self.data_encode)
        
        # Modify the names of the columns with the proper suffix
        self.new_names = []
        for c in self.transformed.columns:
            self.new_names.append(c+'_'+self.encoding_type)
        self.transformed.columns = self.new_names
         
        if self.return_categorical:
            #print('The encoding {} has made {} columns, the input was {} and the output shape{}'.
             #     format(self.encoding_type,self.transformed.shape, X.shape,self.transformed.join(X).shape))
            #print(self.transformed.join(X).dtypes)

            return self.transformed.join(X)
        else:
            return self.transformed.join(X)._get_numeric_data()

class NaNtreatment(BaseEstimator, TransformerMixin):
    '''
    This class implements a fit and transform methods that enables to implace NaNs in different ways.
    '''
    def __init__(self, treatment="mean"):
        self._allowed_treatments = ["fixed_value", "mean",'median','mode','None']     
        assert treatment in self._allowed_treatments or isinstance(treatment,(int,float)),  "the treatment introduced {} is not valid. Please use one in {}".format(treatment, self._allowed_treatments)
        self.treatment = treatment
    
    def fit(self, X, y):
        """
        Learns statistics to impute nans.
        """
        
        if self.treatment == "mean" or self.treatment==None:
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.treatment == "median":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median')
        elif self.treatment == "most_frequent":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif isinstance(self.treatment, (int,float)):
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                                                 strategy="constant",fill_value=self.treatment)       
        

        self.treatment_method.fit(X.values)
        return self

    def transform(self, X):
        if self.treatment==None:
            return X
        return self.treatment_method.transform(X)


def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.  
    From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    WARNING! THIS CAN DAMAGE THE DATA 
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
def auc_score(y_true, preds):
    return sklearn.metrics.roc_auc_score(y_true, preds[:,1])

