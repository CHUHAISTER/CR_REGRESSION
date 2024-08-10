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
import warnings
warnings.filterwarnings("ignore")
# custom files
import columns

# read train data
ds = pd.read_csv("data_train.csv")



feature_engineering_pipeline = Pipeline([
    ('encoder', SklearnTransformerWrapper(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                                          variables=columns.categorical_columns)),  # Ordinal encode categorical variable
    ('scaler', SklearnTransformerWrapper(MinMaxScaler())),  # Standardize variables
])
    


# Function to clean and convert values
def clean_and_convert(value):
    cleaned_value = value.replace('+', '').replace('-', '')
    if cleaned_value:
        return int(cleaned_value)
    else:
        return None

# Apply clean_and_convert function to specific columns
for col in ds.columns[14:20]:  
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

desired_value = '—(immune to damage)'
filtered_ds = ds[ds['hp special'] == desired_value]
index_to_drop = filtered_ds.index
ds.drop(index_to_drop, inplace=True)

desired_value = '127 (17d8+51) reduced to 107; subtract 1 for each day that passes during the adventure'
filtered_indices = ds.index[ds['hp special'] == desired_value].tolist()
column_to_change = 'hp'
new_value = 127
ds.loc[filtered_indices, column_to_change] = new_value

desired_value = '1'
filtered_indices = ds.index[ds['hp special'] == desired_value].tolist()
column_to_change = 'hp'
new_value = 1
ds.loc[filtered_indices, column_to_change] = new_value


def convert_to_decimal(value):
    try:
        numerator, denominator = map(int, value.split('/'))
        return numerator / denominator
    except:
        if(value == "Unknown"):
            return 0
        else:
            return value


ds['cr'] = ds['cr'].apply(convert_to_decimal)
ds['cr'] = pd.to_numeric(ds['cr'], errors='coerce')

columns_to_replace = ['walk',	'fly',	'swim',	'burrow',	'climb'	]  # Список назв стовпців, які потрібно замінити

ds[columns_to_replace] = ds[columns_to_replace].replace('-', 0)
for col in columns_to_replace:
    ds[col] = pd.to_numeric(ds[col], errors='coerce')

ds['hp'] = pd.to_numeric(ds['hp'], errors='coerce')
ds['ac'] = pd.to_numeric(ds['ac'], errors='coerce')

columns_to_drop = ['Unnamed: 0', 'hp formula', 'hp special', 'ac special', ]
for col in columns_to_drop:
        ds = ds.drop(col, axis=1)

ds['hover'] = ds['hover'].replace('-', 'False')

value_counts = ds['alignment'].value_counts()
other_values = value_counts[value_counts <= 10].index
ds['alignment'] = ds['alignment'].replace(other_values, 'other')

value_counts = ds['source'].value_counts()
other_values = value_counts[value_counts <= 10].index
ds['source'] = ds['source'].replace(other_values, 'other')

# Define target and features columns
X = ds[columns.X_columns]
y = ds[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = feature_engineering_pipeline.fit_transform(X_train)

X_test = feature_engineering_pipeline.transform(X_test)
with open('pipeline.pkl', 'wb') as handle:
    pickle.dump(feature_engineering_pipeline, handle)
# Building and train Random Forest Model
rf = RandomForestRegressor(random_state=15)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R-squared:', metrics.r2_score(y_test, y_pred))
filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))
