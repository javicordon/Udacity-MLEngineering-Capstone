import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from math import ceil
import seaborn as sns
import pickle

from config import config

class explore():
    """Exploring dataset.
    """

    def __init__(self, data = None, sep=';', debug=False, th_na_p=0.30, th_na_col=50, th_high_unique=150) -> None:
        """It is expected to get either a pandas dataFrame as input
        or the path location to a csv file containing the data.
        data: a pandas dataFrame or path location to read one.
        sep: separator used in a csv file to read into a pandas df.
        debug: True/False will print out information of what is 
        being processed and some metrics.
        """
        self.debug = debug
        if type(data) is pd.DataFrame:
            self.data = data
        else:
            self.data = pd.read_csv(data, sep=sep)
        
        self.th_na_p = th_na_p
        self.th_na_col = th_na_col
        self.th_high_unique = th_high_unique
        self.dict_outliers_criteria = None
        self.dict_replace_unknown = None
        
        if self.debug:
            print(f"DataFrame has shape: {self.data.shape}")
            
    def encode_na(self, dict_encoder=None):
        """This function uses a catalog given, where a number or string
        represents unknown data or similar. Here it is encoded as NaN,
        given this dictionary.
        """
        if(dict_encoder==None):
            dict_encoder = self.dict_replace_unknown
        self.data = self.data.replace(dict_encoder)
            
    def missing_values(self, th_na_p=None, th_na_col=None, drop_na_rows=False, drop_na_columns=False) -> None:
        if (th_na_p==None):
            th_na_p=self.th_na_p
        if (th_na_col==None):
            th_na_col=self.th_na_col
        # make a list of the variables that contain missing values
        vars_with_na = [var for var in self.data.columns if self.data[var].isnull().sum() > 0]

        # determine percentage of missing values
        self.missing_p = self.data[vars_with_na].isnull().mean()
        
        # filter out variables that have more than th% of missing records
        self.missing_higher_than_th = [var for var in self.data.columns if self.data[var].isnull().mean() > th_na_p]
        if(drop_na_columns):
            self.data.drop(columns=self.missing_higher_than_th, inplace=True)
        elif(len(self.missing_higher_than_th)>0):
            # Find those rows which all columns are nans, among columns with high % of NaN.
            all_rows_na = self.data[self.data[self.missing_higher_than_th].isnull().all(1)].index.values
        
        # Find rows which number of NaN columns is higher than col_na_th
        rows_high_nan = self.data[self.data.isnull().sum(1)>th_na_col].index.values
        if(drop_na_rows):
            self.data.drop(index=rows_high_nan, axis=1, inplace=True)

        
        if self.debug:
            print(f"Number of columns with missinge values higher than th_na_p {th_na_p}: {len(self.missing_higher_than_th)} {round(len(self.missing_higher_than_th)*100/self.data.shape[1],2)}% ")
            print(f"Number of rows which number of NaN columns is higher than {th_na_col}: {len(rows_high_nan)} {round(len(rows_high_nan)*100/self.data.shape[0],2)}%")
            if(drop_na_columns):
                print(f"Shape after dropping columns with high number of NaN: {self.data.shape}")
            elif(len(self.missing_higher_than_th)>0):
                print(f"Number of rows which all rows are NaN, from columns higher than th_na_p {th_na_p}: {len(all_rows_na)} {round(len(all_rows_na)*100/self.data.shape[0],2)}%")
        return
    
    def detailed_columns(self, th_high_unique=None, drop=False) -> None:
        if(th_high_unique==None):
            th_high_unique = self.th_high_unique
            
        self.high_unique = {col:len(self.data[col].unique()) for col in self.data.columns.to_list() if (len(self.data[col].unique()) > th_high_unique)}
        if(self.debug):
            print(f"Number of columns where unique values are higher than {th_high_unique}: {len(self.high_unique)}")
            
        if(drop):
            self.data.drop(columns=list(self.high_unique.keys()), inplace=True)
            if(self.debug):
                print(f"Data shape after droping high detailed columns: {self.data.shape}")
        
        return
    
    def numerical_columns(self) -> None:
        # make list of numerical variables
        self.num_vars = [var for var in self.data.columns if self.data[var].dtypes != 'O']

        # make list of discrete variables
        self.discrete_vars = [var for var in self.data.columns if self.data[var].dtypes == 'int64']

        # make list of continuous variables
        self.cont_vars = [
            var for var in self.num_vars if var not in self.discrete_vars]
        
        # Get columns that are discrete but contains only int values
        self.cont_is_discrete = []
        for col in self.cont_vars:
            uniques = self.data[col].value_counts().index.values
            if(~np.array([False for val in uniques if val.is_integer()]).any()):
                self.cont_is_discrete.append(col)

        if self.debug:
            print('Number of numerical variables: ', len(self.num_vars))
            print('Number of discrete variables: ', len(self.discrete_vars))
            print('Number of continuous variables: ', len(self.cont_vars))
        
        return
    
    def categorical_columns(self, th_cat = 45) -> None:
        # capture categorical variables in a list
        self.cat_vars = [var for var in self.data.columns if self.data[var].dtypes == 'O']

        

        self.high_categories = [var for var in self.cat_vars if self.data[var].nunique() > th_cat]
        
        if self.debug:
            print('Number of categorical variables: ', len(self.cat_vars))
            print(f"Number of columns where number of categories is higher than th_cat {th_cat}: {len(self.high_categories)}")
        
        return
        
    def columns_dtypes(self) -> None:
        self.numerical_columns()
        self.categorical_columns()
        return
    
    def columns_stats(self) -> None:
        self.missing_values()
        self.columns_dtypes()
        
    def find_outliers(self, z_th=2.33) -> None:
        """Find outliers across all columns.
        z_th: is the number of standard deviatons to use
        as a treshold, to select which rows will be considered
        outliers. You can use a standard normal distribution
        table, to select what % of the normal distribution
        you want to cover.
        """
        
        z_dict_outliers = {}
        z_dict_non_outliers = {}
        
        # Find iterating each column
        for var in self.num_vars:
            df_var = self.data.loc[self.data[var].notna(), var].to_frame()
            z = np.abs(stats.zscore(df_var))
            outliers = np.where(z > z_th)
            
            # If zscore lower than z_th then no outliers by criteria
            #  Saves the mean, median and standard deviaton per column.
            #  Also keeps a missing percentage value per column.
            if (outliers[0].shape[0] == 0):
                z_mean = df_var[var].mean()
                z_median = df_var[var].median()
                z_std = df_var[var].std()
                z_dict_non_outliers[var] = {
                    'mean': z_mean, 'median': z_median, 'std': z_std,
                    'missing_p': (1-(df_var.shape[0]/self.data.shape[0]))} #'df': df_var
                continue
                
            # When column has outliers, discard them from mean and median calc.
            #  Keep a list of those outliers to work on later. Either by dropping
            #  or replacing by a criteria value.
            df_var[var+'_z'] = z
            z_mean = df_var.loc[df_var[var+'_z']<=z_th, var].mean()
            z_median = df_var.loc[df_var[var+'_z']<=z_th, var].median()
            z_std = df_var.loc[df_var[var+'_z']<=z_th, var].std()
            z_outliers = df_var.loc[df_var[var+'_z']>z_th, var].index.to_list()
            z_dict_outliers[var]= {'mean': z_mean, 'median': z_median, 'std': z_std,
                          'outliers': z_outliers, 'outliers_p': len(z_outliers)/df_var.shape[0],
                          'outliers_p_abs': len(z_outliers)/self.data.shape[0], 
                          'missing_p': (1-(df_var.shape[0]/self.data.shape[0]))}
            
        # Sometimes the same records share outliers across different columns
        #     find the columns which number of outliers percentage are the same.
        outliers_percentages = pd.Series([z_dict_outliers[x]['outliers_p'] for x in z_dict_outliers.keys()]).value_counts()
        repeated_outliers_percentages = outliers_percentages.loc[outliers_percentages>1]
        non_repeated_outliers_percentages = outliers_percentages.loc[outliers_percentages==1]

        # Get rows/records that match being outliers in different columns
        repeated_outliers_records = []
        for percentage, number_of_features in repeated_outliers_percentages.iteritems():
            rep_out = ([x for x in z_dict_outliers.keys() if round(z_dict_outliers[x]['outliers_p'],6)==round(percentage,6)])
            outliers_num = len(set(np.array([z_dict_outliers[x]['outliers'] for x in rep_out]).reshape(1,-1)[0]))
            matching_feat_count = (pd.Series(np.array([z_dict_outliers[x]['outliers'] for x in rep_out]).reshape(1,-1)[0]).value_counts()==number_of_features)
            match = matching_feat_count.sum()
            repeated_outliers_records = repeated_outliers_records+matching_feat_count.keys().to_list()

            if self.debug:
                print(f"Number of Features matching same outliers percentage records: {number_of_features}")
                print(f"\tNumber of unique outliers records: {outliers_num}")
                print(f"\tNumber of records repeated on all {number_of_features} features: {match}")
                print(f"\tPercentag of the records with outliers vs total: {outliers_num/self.data.shape[0]}")
                print(f"\tColumns: {rep_out} \n")            

        self.repeated_outliers_percentages = repeated_outliers_percentages
        self.non_repeated_outliers_percentages = non_repeated_outliers_percentages
        # Get rows/records of outliers on the rest of column
        non_repeated_outliers_records = []
        non_repeated_outliers_features = []
        for percentage, number_of_features in non_repeated_outliers_percentages.iteritems():
            rep_out = ([x for x in z_dict_outliers.keys() if round(z_dict_outliers[x]['outliers_p'],6)==round(percentage,6)])
            outliers_num = len(set(np.array([z_dict_outliers[x]['outliers'] for x in rep_out]).reshape(1,-1)[0]))
            matching_feat_count = (pd.Series(np.array([z_dict_outliers[x]['outliers'] for x in rep_out]).reshape(1,-1)[0]).value_counts()==number_of_features)
            non_repeated_outliers_records = non_repeated_outliers_records+matching_feat_count.keys().to_list()
            non_repeated_outliers_features = non_repeated_outliers_features + rep_out
            if self.debug:
                print(f"Column: {rep_out}")
                print(f"\tNumber of unique outliers records: {outliers_num}")
                print(f"\tPercentag of the records with outliers vs total: {outliers_num/self.data.shape[0]} \n")
                
        self.repeated_outliers_percentages = repeated_outliers_percentages
        self.non_repeated_outliers_percentages = non_repeated_outliers_percentages
        self.non_repeated_outliers_records = non_repeated_outliers_records
        self.non_repeated_outliers_features = non_repeated_outliers_features
        self.z_dict_outliers = z_dict_outliers
        self.z_dict_non_outliers = z_dict_non_outliers
        
        return
    
    def remove_noisy_columns(self, th_noise=0.15) -> None:
        """Where noise is the percentage of outliers +
        percentage of missing for one column. We will
        remove those whose percentage is higher than
        th_noise.
        """
        feats_high_noise = []
        for feat in self.non_repeated_outliers_features:
            outlier_p = self.z_dict_outliers[feat]['outliers_p_abs']
            missing_p = self.z_dict_outliers[feat]['missing_p']
            if(outlier_p+missing_p>th_noise):
                self.z_dict_outliers.pop(feat)
                feats_high_noise = feats_high_noise + [feat]
                if self.debug: 
                    print(f"Feature {feat}, has a percentage of outliers+missing: {outlier_p+missing_p}")
                    
        #Drop noisy columns
        self.feats_high_noise = feats_high_noise
        self.data = self.data.drop(columns=feats_high_noise)
        
        return
    
    def z_dict_replacer(self, z_th=2.33) -> None:
        z_dict_replacer = {}
        for var in self.z_dict_outliers.keys():
            df_var = self.data.loc[self.data[var].notna(), var].to_frame()
            z = np.abs(stats.zscore(df_var))
            outliers = np.where(z > z_th)
            df_var[var+'_z'] = z
            z_mean = df_var.loc[df_var[var+'_z']<=z_th, var].mean()
            z_median = df_var.loc[df_var[var+'_z']<=z_th, var].median()
            z_std = df_var.loc[df_var[var+'_z']<=z_th, var].std()
            z_outliers = df_var.loc[df_var[var+'_z']>z_th, var].index.to_list()
            z_dict_replacer[var] = {'mean': z_mean, 'median': z_median, 'std': z_std,
                          'outliers': z_outliers, 'outliers_p': len(z_outliers)/df_var.shape[0],
                          'outliers_p_abs': len(z_outliers)/self.data.shape[0], 
                          'missing_p': (1-(df_var.shape[0]/self.data.shape[0]))}
            
        self.z_dict_outliers_replacer = z_dict_replacer
        
        self.dict_numerical_encoder()
        
        return
    
    
    def dict_numerical_encoder(self) -> None:
        dict_numerical = {}
        
        dict_numerical.update({k:{'mean': self.z_dict_outliers_replacer[k]['mean'], 
                                 'median': self.z_dict_outliers_replacer[k]['median'],
                                 'std': self.z_dict_outliers_replacer[k]['std']} for k in self.z_dict_outliers_replacer})
        dict_numerical.update({k:{'mean': self.z_dict_non_outliers[k]['mean'], 
                                 'median': self.z_dict_non_outliers[k]['median'],
                                 'std': self.z_dict_non_outliers[k]['std']} for k in self.z_dict_non_outliers})

        self.dict_numerical = dict_numerical
            
        return
    
    def dict_categorical(self) -> None:
        dict_categorical_encoder = {}
        for col in self.cat_vars:
            dict_categorical_encoder[col] = {key: i for i, key in enumerate(self.data[col].value_counts().index)}
        
        self.dict_categorical_encoder = dict_categorical_encoder
        return
    
    def remove_non_variant_columns(self) -> None:
        
        
        features = [self.dict_numerical, self.dict_categorical_encoder]
        for dct in features:
            remove_col = []
            for column_name in dct.keys():
                unique_values = self.data[column_name].value_counts().shape[0]
                if(unique_values==1):
                    remove_col.append(column_name)
                    if(self.debug): print(f"Non variant column: {column_name}")
            for key in remove_col:
                dct.pop(key)   
            self.data = self.data.drop(remove_col, axis=1)
        
        return
    
    def remove_columns(self, columns) -> None:
        
        for col in columns:
            try:
                self.dict_categorical_encoder.pop(col)
            except:
                self.dict_numerical.pop(col)
        
        self.data.drop(columns=columns, inplace=True)
        return

    def replace_NA(self, dict_criteria=None) -> None:
        if self.dict_numerical==None:
            #self.z_dict_replacer()
            raise Exception('No dict_numerical found')
        if self.dict_categorical_encoder==None:
            #self.dict_categorical()
            raise Exception('No dict_categorical found')
            
        # replace na by the median, in columns numerical columns
        for feat in self.dict_numerical.keys():
            self.data.loc[self.data[feat].isna(), feat] = self.dict_numerical[feat]['median']

        # replace na by the most common category in categorical columns
        #cat_vars = self.dict_categorical_encoder.keys()
        #self.data[cat_vars] = self.data[cat_vars].fillna(0)
        for feat in self.dict_categorical_encoder.keys():
            self.data.loc[self.data[feat].isna(), feat] = 0
            
        return
    
    def replace_outliers(self, dict_criteria=None, th_std = 2.33) -> None:
        
        if(dict_criteria==None):
            dict_criteria = self.dict_outliers_criteria
        
        def get_outliers(column):
            z_mean = self.dict_numerical[column]['mean']
            z_std = self.dict_numerical[column]['std']
            df_z = self.data[[column]].copy()
            df_z[column+'_z'] = np.abs((self.data[column]-z_mean)/z_std) > th_std
            return df_z[column+'_z']
        
        def max(column):
            z_mean = self.dict_numerical[column]['mean']
            z_std = self.dict_numerical[column]['std']
            std_max = ceil(z_mean+th_std*z_std)
            outliers = get_outliers(column)
            self.data.loc[outliers, column] = std_max
            
        def mean(column):
            outliers = get_outliers(column)
            self.data.loc[outliers, column] = self.dict_numerical[column]['mean']
            
        def median(column):
            outliers = get_outliers(column)
            self.data.loc[outliers, column] = self.dict_numerical[column]['median']
        
        def menu(column, option):
            options = {
                'max': max,
                'mean': mean,
                'median': median,
            }
            return options[option](column)
            
        for column in dict_criteria:
            menu(column, dict_criteria[column])
        
        return
    
    def replace_numerical_outliers(self) -> None:
        if self.z_dict_outliers_replacer==None:
            #self.z_dict_replacer()
            raise Exception('No z_dict_outliers_replacer found')
        
        # here goes the magic and replace outliers by the median
        for feat in self.z_dict_outliers_replacer.keys():
            self.data.loc[self.z_dict_outliers_replacer[feat]['outliers'], feat] = self.z_dict_outliers_replacer[feat]['median']
        
        return
    
    def replace_categorical(self) -> None:
        if self.dict_categorical_encoder==None:
            #self.dict_categorical()
            raise Exception('No dict_categorical found')
            
        # replace with categorical encoder dictionary form previous step
        self.data = self.data.replace(self.dict_categorical_encoder)
        
        # Replace unseen categories by most common category (0)
        for col in self.dict_categorical_encoder.keys():
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0).astype('int64')

        return
    
    def encode_categorical_numbers(self, convert_inplace=False) -> None:
        # Get categorical columns that contain only numbers
        categorical_is_number = []
        for col in self.cat_vars:
            uniques = self.data[col].value_counts().index.values
            try:
                if(~np.array([False for val in uniques if isinstance(float(val), float)]).any()):
                    categorical_is_number.append(col)
            except:
                continue
        
        if(convert_inplace):
            self.data[categorical_is_number] = self.data[categorical_is_number].astype(float)
        self.categorical_is_number = categorical_is_number
        
        return
    
    def encode_numerical_to_float(self) -> None:
        # Set to int those float columns which all values are int
        self.data[self.num_vars] = self.data[self.num_vars].astype(float)
        
    def normalize(self, keep_cols=[]) -> None:
        self.th_na_p=config.TH_NA_P
        self.th_na_col=config.TH_NA_COL
        self.th_high_unique=config.TH_HIGH_UNIQUE
        self.dict_categorical_encoder = config.dict_categorical_encoder
        self.dict_numerical = config.dict_numerical
        self.dict_outliers_criteria = config.dict_outliers_criteria
        self.dict_replace_unknown = config.dict_replace_unknown
        keep_cols = keep_cols+config.FEATURES
        
        if(self.debug): print('Start Normalizing')
        self.columns_stats()
        self.encode_numerical_to_float()
        if(self.debug): print('1 - Done encoding Numerical to Float')
        self.encode_na()
        if(self.debug): print('2 - Done encoding NaNs by catalog')
        self.encode_categorical_numbers(convert_inplace=True)
        if(self.debug): print('3 - Done encoding categorical numbers to Float')
        self.replace_categorical()
        if(self.debug): print('4 - Done encoding categorical strings to Numbers')
        self.replace_outliers()
        if(self.debug): print('5 - Done replacing Outliers')
        self.replace_NA()
        self.data = self.data[keep_cols]
        # Verify there are no NaNs
        vars_with_nan = [var for var in self.data.columns if self.data[var].isnull().sum() > 0]
        if(len(vars_with_nan)>0):
            print('Error normalizing. Variables contain NaNs:', vars_with_nan)
        if(self.debug): print('6 - Done validating no NaNs')
        if(self.debug): print('End Normalizing')
        if(self.debug): print(f"Final data shape: {self.data.shape}")
            
        
        return self
    
class nan_encoder():
    """Exploring dataset.
    """

    def __init__(self, data = None, sep=';',
                 col_to_detect_nan='meaning',
                 string_to_detect_nan='unknown',
                 column_with_features='attribute',
                 col_with_values_of_nan='value',
                 debug=False) -> None:
        """It is expected to get either a pandas dataFrame as input
        or the path location to a csv file containing the data.
        data: a pandas dataFrame or path location to read one.
        sep: separator used in a csv file to read into a pandas df.
        debug: True/False will print out information of what is 
        being processed and some metrics.
        """
        self.debug = debug
        self.col_to_detect_nan = col_to_detect_nan
        self.string_to_detect_nan = string_to_detect_nan
        self.column_with_features = column_with_features
        
        if type(data) is pd.DataFrame:
            self.data = data
        else:
            self.data = pd.read_csv(data, sep=sep)
        
        self.unknown_df = None
        
        if self.debug:
            print(f"DataFrame has shape: {self.data.shape}")
            

    
    def get_df(self):
        def cast_int(x):
            x = x.split(',')
            new_x = []
            for i in x:
                try:
                    new_x.append(float(i))
                except:
                    new_x.append(i)
            return new_x
    
        #Transform strings to floats
        unknown_df = self.data.loc[self.data[self.col_to_detect_nan]==self.string_to_detect_nan].copy().reset_index()
        unknown_df['value']=unknown_df['value'].apply(lambda x: cast_int(x))
        self.unknown_df = unknown_df
        
        return
    
    def get_dict(self):
        # Make a Dictionary for replacing encoded NaNs
        self.get_df()
        unknown_dict = self.unknown_df[[self.column_with_features,'value']].set_index(self.column_with_features).to_dict()['value']
        unknown_replace_dict = {key:{value:np.nan} for key in unknown_dict for value in unknown_dict[key]}
        {unknown_replace_dict[key].update({value:np.nan}) for key in unknown_dict for value in unknown_dict[key]}
        self.unknown_replace_dict = unknown_replace_dict
        
        return unknown_replace_dict
    
        
