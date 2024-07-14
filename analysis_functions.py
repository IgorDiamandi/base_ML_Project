import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from data_functions import compute_statistics, replace_outliers, \
    apply_one_hot_encoder, unite_sparse_columns, missing_values_imputation, \
    split_fiProductClassDesc
from model_functions import train_and_evaluate_model

# Constants
LEVEL_OF_PARALLELISM = -1
NUMBER_OF_TREES = [20]  # For grid search
TREE_DEPTH = [10, 20, 30]  # For grid search
MIN_SAMPLES_SPLIT = [2, 5, 10]  # For grid search
MIN_SAMPLES_LEAF = [1, 2, 4]  # For grid search
MAX_FEATURES = ['auto', 'sqrt', 'log2']  # For grid search

KILL_THEM_COLUMNS = ['uControl_any_non_null', 'uMounting_any_non_null', 'uHydraulics_non_null_count',
                     'uControl_non_null_count', 'fiProductClassDesc']

SIZE_FIT_COLUMNS = ['fiModelDesc', 'post_fi_Horsepower', 'Drive_System', 'Stick_Length', 'Undercarriage_Pad_Width']

USAGE_FIT_COLUMNS = ['age', 'MachineHoursCurrentMeter']

DUPLICATE_COLUMNS = ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 'fiModelDescriptor']
COLUMN_GROUPS = {
    'uBladeStick': ['Blade_Extension', 'Stick_Length', 'Stick'],
    'uTrack': ['Pad_Type', 'Grouser_Type', 'Grouser_Tracks'],
    'uMounting': ['Backhoe_Mounting', 'Forks', 'Pushblock', 'Ripper', 'Scarifier', 'Thumb'],
    'uControl': ['Travel_Controls', 'Ride_Control', 'Transmission',
                 'Pattern_Changer', 'Tip_Control', 'Coupler', 'Coupler_System'],
    'uHydraulics': ['Turbocharged', 'Hydraulics', 'Hydraulics_Flow'],
    'uDrive': ['Differential_Type', 'Drive_System']
}

DTYPE_SPEC = {
    13: 'str',
    39: 'str',
    40: 'str',
    41: 'str'
}

df = pd.read_csv('Data\\train.csv', dtype=DTYPE_SPEC)
df_valid = pd.read_csv('Data\\valid.csv', dtype=DTYPE_SPEC)

# Product size handling
df, df_valid = split_fiProductClassDesc([df, df_valid], 'fiProductClassDesc')

df, df_valid = missing_values_imputation([df, df_valid], 'ProductSize', SIZE_FIT_COLUMNS)

# Grouping columns with technical details
for new_column, columns in COLUMN_GROUPS.items():
    df, df_valid = unite_sparse_columns([df, df_valid], columns, new_column)

# Drop unimportant columns
df.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)
df_valid.drop(KILL_THEM_COLUMNS, axis=1, inplace=True)

# Drop duplicated columns
df.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)
df_valid.drop(DUPLICATE_COLUMNS, axis=1, inplace=True)

# Convert dates to periods
df['saleyear'] = pd.to_datetime(df['saledate']).dt.year
df_valid['saleyear'] = pd.to_datetime(df_valid['saledate']).dt.year

df['salemonth'] = pd.to_datetime(df['saledate']).dt.month
df_valid['salemonth'] = pd.to_datetime(df_valid['saledate']).dt.month

df['saledayofweek'] = pd.to_datetime(df['saledate']).dt.dayofweek
df_valid['saledayofweek'] = pd.to_datetime(df_valid['saledate']).dt.dayofweek

df['age'] = 2024 - df['YearMade']
df_valid['age'] = 2024 - df_valid['YearMade']

df['AgeAtSale'] = df['saleyear'] - df['YearMade']
df_valid['AgeAtSale'] = df_valid['saleyear'] - df_valid['YearMade']

# Drop unnecessary columns
df.drop(['saledate', 'YearMade'], axis=1, inplace=True)
df_valid.drop(['saledate', 'YearMade'], axis=1, inplace=True)

# Usage band handling
df, df_valid = missing_values_imputation([df, df_valid], 'UsageBand', USAGE_FIT_COLUMNS)

# Convert 'MachineID' column into 'TimesAppearing' column
df['TimesAppearing'] = df['MachineID'].map(df['MachineID'].value_counts())
df.drop('MachineID', axis=1, inplace=True)
df_valid['TimesAppearing'] = df_valid['MachineID'].map(df_valid['MachineID'].value_counts())
df_valid.drop('MachineID', axis=1, inplace=True)

# Replace NaN with 0 in 'ID' columns
df[['datasource', 'auctioneerID']] = df[['datasource', 'auctioneerID']].fillna(0.0)
df_valid[['datasource', 'auctioneerID']] = df_valid[['datasource', 'auctioneerID']].fillna(0.0)

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.3, random_state=42)

# Get statistics dataframe using compute_statistics function from the X_train
statistics_df = compute_statistics(X_train)

# Use replace_outliers function to replace outliers
X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'MachineHoursCurrentMeter', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'AgeAtSale', 'iqr', statistics_df)

X_train, X_test, df_valid = replace_outliers([X_train, X_test, df_valid],
                                             'age', 'iqr', statistics_df)

X_train.to_csv('Data\\X_train.csv', index=False)

# Apply one-hot encoding
X_train_transformed, X_test_transformed, X_valid_transformed = (
    apply_one_hot_encoder([X_train, X_test, df_valid],
                          X_train.select_dtypes(exclude=['number']).columns.tolist()))

# Define preprocessing steps
categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()

# Define preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creating the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define the parameter grid for grid search
param_grid = {
    'regressor__n_estimators': NUMBER_OF_TREES,
    'regressor__max_depth': TREE_DEPTH,
    'regressor__min_samples_split': MIN_SAMPLES_SPLIT,
    'regressor__min_samples_leaf': MIN_SAMPLES_LEAF,
    'regressor__max_features': MAX_FEATURES
}

# Perform Grid Search with cross-validation using multithreading
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=LEVEL_OF_PARALLELISM, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)

print(f'Best parameters: {best_params}')
print(f'Best cross-validation RMSE: {best_score}')

# Use the best parameters to create a new model
print(grid_search.best_estimator_)