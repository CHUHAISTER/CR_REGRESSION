import pickle
import math 
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings("ignore")

# custom files
import columns

# read train data
ds = pd.read_csv("data_test.csv")
print('new data size', ds.shape)

with open('pipeline.pkl', 'rb') as handle:
    loaded_pipeline = pickle.load(handle)
    

# Function to clean and convert values
def clean_and_convert(value):
    cleaned_value = value.replace('+', '').replace('-', '')
    if cleaned_value:
        return int(cleaned_value)
    else:
        return None

# Apply clean_and_convert function to specific columns
for col in ds.columns[8:14]:  
    ds[col] = ds[col].apply(clean_and_convert)

# Function to round down values
def round_down(x):
    return math.floor(x)

# Function to fill missing values
def fill_missing(df, stat):
    df[stat] = df[stat].astype(int)
    x = (df[stat] - 10) / 2
    x = x.apply(round_down)
    df[stat + " save"].fillna(x, inplace=True)

# List of stats columns
stat_list = ['str', 'dex', 'con', 'int', 'wis', 'cha']

# Apply fill_missing function to each stat
for stat in stat_list:
    fill_missing(ds, stat)




columns_to_replace = ['walk',	'fly',	'swim',	'burrow',	'climb'	]  # Список назв стовпців, які потрібно замінити

ds[columns_to_replace] = ds[columns_to_replace].replace('-', 0)
for col in columns_to_replace:
    ds[col] = pd.to_numeric(ds[col], errors='coerce')

ds['hp'] = pd.to_numeric(ds['hp'], errors='coerce')
ds['ac'] = pd.to_numeric(ds['ac'], errors='coerce')

ds['hover'] = ds['hover'].replace('-', 'False')

value_counts = ds['alignment'].value_counts()
other_values = value_counts[value_counts <= 10].index
ds['alignment'] = ds['alignment'].replace(other_values, 'other')

value_counts = ds['source'].value_counts()
other_values = value_counts[value_counts <= 10].index
ds['source'] = ds['source'].replace(other_values, 'other')

# Define target and features columns
X = ds[columns.X_columns]
X_transformed = loaded_pipeline.transform(X)

# load the model and predict
rf = pickle.load(open('finalized_model.sav', 'rb'))

y_pred = rf.predict(X_transformed)

ds['CR_pred'] = rf.predict(X_transformed)
ds.to_csv('prediction_results.csv', index=False)
